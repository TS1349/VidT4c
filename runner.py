import os
import random
import torch.multiprocessing as mp
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
# from torchaudio.transforms import Spectrogram

from torchvision.transforms.v2 import Compose, Normalize, Resize, ToDtype, RandomHorizontalFlip, ColorJitter
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from world_info import init_distributed_mode
from trainer import PTrainer
from models import BridgedTimeSFormer4C, BridgedViViT4C, BridgedVideoSwin4C, AudioVisionTransformer, TVLTTransformer
from dataloader import EAVDataset, EmognitionDataset, MDMERDataset
from dataloader.transforms import AbsFFT, STFT
from scheduler import CosineScheduler

from torch.nn.parallel import DistributedDataParallel as DDP

import argparse

def get_torch_model(name):
    if name == "vivit":
        return BridgedViViT4C
    elif name == "tsf":
        return BridgedTimeSFormer4C
    elif name == "swin":
        return BridgedVideoSwin4C
    elif name == "hicmae":
        return AudioVisionTransformer
    elif name == "tvlt":
        return TVLTTransformer
    else:
        raise Exception("Wrong model name")

def get_torch_dataset(name):
    if name == "mdmer":
        return MDMERDataset
    elif name == "emognition":
        return EmognitionDataset
    elif name == "eav":
        return EAVDataset
    else:
        raise Exception("Wrong dataset name")


def _load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0.0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            lr_scale = self.current_step / self.warmup_steps
        else:
            progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr_scale = 0.5 * (1 + math.cos(math.pi * progress))
        for param_group in self.optimizer.param_groups:
            base_lr = param_group.get("initial_lr", param_group["lr"])
            param_group["lr"] = self.min_lr + (base_lr - self.min_lr) * lr_scale

def run(rank, args):
    print(f"[RANK {rank}] Starting process...")

    # Minimize randomness
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Recall args option
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    csv_file = args.csv_file
    experiment_name = args.experiment_name
    checkpoint_dir =args.checkpoint_dir
    torch_model = get_torch_model(args.model)
    fft_mode = args.fft_mode
    img_size = args.img_size

    torch_dataset = get_torch_dataset(args.dataset)

    print(f"dataset: {torch_dataset}")
    print(f"model: {torch_model}")

    # split process group_intialization
    idr_torch = None
    if ("j_zay" == args.server):

        idr_torch=init_distributed_mode()

        if(idr_torch is not None):
            world_size = idr_torch.size
            rank = idr_torch.rank
            local_rank = idr_torch.local_rank
            num_replicas = idr_torch.size
        else:
            print("couldn't initialize idr_torch")
            return -1

    elif ("nef" == args.server):
        world_size = args.num_gpus
        rank = rank
        local_rank = rank
        num_replicas = args.num_gpus
    else:
        print("This point should be unreachable")
        return -2

    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            world_size=world_size,
                            rank=rank)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    else:
        print("No GPU available")
 
    # Data processing
    video_preprocessor_train = Compose([
        Resize(size=(img_size, img_size), antialias=True),
        RandomHorizontalFlip(p=0.5),
        ColorJitter(brightness=0.2, contrast=0.2),
        lambda x: x.float().div(255.0),
        Normalize(mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225)),
    ])

    # During infernece, we don't need other transformations.
    video_preprocessor_val = Compose([
        Resize(size=(img_size, img_size), antialias=True),
        lambda x: x.float().div(255.0),
        Normalize(mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225)),
    ])

    # Check the right fourier transformation
    fft = None
    if fft_mode == "AbsFFT":
        fft = AbsFFT(dim=-2)
        freq_bin = 64
    elif fft_mode == "Spectrogram":
        fft = STFT()
        freq_bin = 128 * 256

    training_dataset = torch_dataset(
        csv_file=csv_file,
        time_window = 15.0, #sec -> change to avoid duplication of eeg sampling
        video_transform=video_preprocessor_train,
        eeg_transform = fft,
        split = "train"
    )
    validation_dataset = torch_dataset(
        csv_file=csv_file,
        time_window = 15.0, #sec
        video_transform=video_preprocessor_val,
        eeg_transform = fft,
        split = "test"
    )

    training_sampler = DistributedSampler(
        dataset=training_dataset,
        num_replicas=num_replicas,
        rank=rank,
    )

    # Add validation sampler
    # validation_sampler = DistributedSampler(
    #     dataset=validation_dataset,
    #     num_replicas=num_replicas,
    #     rank=rank,
    # )

    # We don't need shuffle when using DistributedSampler
    training_dataloader = DataLoader(
        dataset=training_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        sampler=training_sampler,
        drop_last=True, # Add drop_last
    )

    # Val-dataloader shouldn't be sampled by training_sampler
    validation_dataloader = DataLoader(
        dataset=validation_dataset,
        num_workers=4,
        batch_size=batch_size,
        pin_memory=True,
        #sampler=validation_sampler,
        drop_last=False,
    )


    if args.model =='hicmae':
        output_dim = training_dataset.output_shape
        model = torch_model(output_dims=output_dim, all_frames=32, eeg_channels = training_dataset.eeg_channel_count) # Should get num_class from the dataloader
        pretrained_path = os.path.join(os.getcwd(), 'pretrained', args.model + '.pth')
        print(f'Loading pretrained weights from {pretrained_path}')
        checkpoint = torch.load(pretrained_path)

        checkpoint_model = None
        model_key = 'model|module'
        num_frames = 32

        for model_key in model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding (video encoder)
        if 'encoder.pos_embed' in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model['encoder.pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1] # channel dim
            num_patches = model.encoder.patch_embed.num_patches #
            num_extra_tokens = model.encoder.pos_embed.shape[-2] - num_patches # 0/1

            # height (== width) for the checkpoint position embedding
            orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens)//(num_frames // model.encoder.patch_embed.tubelet_size)) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int((num_patches // (num_frames // model.encoder.patch_embed.tubelet_size) )** 0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                # B, L, C -> BT, H, W, C -> BT, C, H, W
                pos_tokens = pos_tokens.reshape(-1, num_frames // model.encoder.patch_embed.tubelet_size, orig_size, orig_size, embedding_size)
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, num_frames // model.encoder.patch_embed.tubelet_size, new_size, new_size, embedding_size)
                pos_tokens = pos_tokens.flatten(1, 3) # B, L, C
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model['encoder.pos_embed'] = new_pos_embed

        # interpolate position embedding (audio encoder), NOTE: assume only time diff!!!
        if 'encoder_audio.pos_embed' in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model['encoder_audio.pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = model.encoder_audio.patch_embed.num_patches
            num_extra_tokens = model.encoder_audio.pos_embed.shape[-2] - num_patches
            # height (!= width) for the checkpoint position embedding
            freq_size = model.encoder_audio.patch_embed.patch_hw[1] # assert the freq dim is fixed (i.e., 128//16=8)
            orig_temporal_size = (pos_embed_checkpoint.shape[-2] - num_extra_tokens) // freq_size
            # height (!= width) for the new position embedding
            new_temporal_size = model.encoder_audio.patch_embed.patch_hw[0]
            # class_token and dist_token are kept unchanged
            if orig_temporal_size != new_temporal_size: # assert the freq dim is fixed (i.e., 128//16=8)
                print("Position (audio) interpolate from %dx to %dx in the temporal dimension" % (
                orig_temporal_size, new_temporal_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(-1, orig_temporal_size, freq_size, embedding_size) # .permute(0, 3, 1, 2)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model['encoder_audio.pos_embed'] = new_pos_embed

        _load_state_dict(model, checkpoint_model)

    elif args.model == 'tvlt':
        model = torch_model(args, output_dim = training_dataset.output_shape, eeg_channels = training_dataset.eeg_channel_count, img_size=img_size, frames = 32)
        pretrained_path = os.path.join(os.getcwd(), 'pretrained', args.model + '.ckpt')
        print(f'Loading pretrained weights from {pretrained_path}')

        checkpoint = torch.load(pretrained_path)
        checkpoint_model = checkpoint.get('model', checkpoint.get('module', checkpoint))

        for k in ['classifier.4.weight', 'classifier.4.bias']:
            if k in checkpoint_model and k in model.state_dict():
                if checkpoint_model[k].shape != model.state_dict()[k].shape:
                    print(f"Removing key {k} due to shape mismatch")
                    del checkpoint_model[k]

        if 'pos_embed_v' in checkpoint_model:
            pos_embed = checkpoint_model['pos_embed_v']
            embedding_size = pos_embed.shape[-1]
            num_patches = model.patch_embed_v.num_patches
            orig_size = int(pos_embed.shape[1] ** 0.5)
            new_size = int(num_patches ** 0.5)
            if orig_size != new_size:
                print(f"Interpolating video pos_embed from {orig_size}x{orig_size} to {new_size}x{new_size}")
                pos_tokens = pos_embed[0].reshape(1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = F.interpolate(pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, new_size * new_size, embedding_size)
                checkpoint_model['pos_embed_v'] = pos_tokens

        if 'pos_embed_a' in checkpoint_model:
            pos_embed = checkpoint_model['pos_embed_a']
            embedding_size = pos_embed.shape[-1]
            target_len = model.max_audio_patches
            if pos_embed.shape[1] != target_len:
                print(f"Interpolating audio pos_embed from {pos_embed.shape[1]} to {target_len}")
                pos_tokens = pos_embed.permute(0, 2, 1)
                pos_tokens = F.interpolate(pos_tokens, size=(target_len,), mode='linear', align_corners=False)
                checkpoint_model['pos_embed_a'] = pos_tokens.permute(0, 2, 1)

        _load_state_dict(model, checkpoint_model)

    else:
        model = torch_model(args,
                    output_dim = training_dataset.output_shape,
                    image_size = img_size,
                    eeg_channels = training_dataset.eeg_channel_count,
                    frequency_bins = freq_bin,
        )
        
        if args.pretrained:
            pretrained_path = os.path.join(os.getcwd(), 'pretrained', args.model + '.pth') # ViViT, TimeSFormer
            print(f'Loading pretrained weights from {pretrained_path}')
            checkpoint = torch.load(pretrained_path)
            model.model.load_state_dict(checkpoint, strict=False) # Backbone is defined as the model.model

    device = torch.device(f"cuda:{rank}")
    model.to(device)
    model = DDP(model, device_ids=[rank])

    loss_function = nn.CrossEntropyLoss()
    loss_function.to(device)

    optimizer = optim.AdamW(
        params=model.parameters(),
        lr=learning_rate, eps=1e-8, betas=(0.9, 0.98),
        weight_decay=weight_decay
)

    steps_per_epoch = math.ceil(len(training_dataset) / batch_size)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(0.1 * total_steps)  # Choose apporopriate warmup step

    for param_group in optimizer.param_groups:
        param_group["initial_lr"] = learning_rate

    # Utlize warmup schedular
    lr_scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr=learning_rate * 0.1,
    )

    # lr_scheduler = CosineScheduler(
    #     optimizer=optimizer,
    #     t_total=total_steps,
    # )

    p_trainer = PTrainer(
        model=model,
        lr_scheduler=lr_scheduler,
        training_dataloader=training_dataloader,
        validation_dataloader=validation_dataloader,
        optimizer=optimizer,
        loss_function=loss_function,
        checkpoint_dir=checkpoint_dir,
        gpu=device,
        gpu_id=rank,
        experiment_name=experiment_name,
    )

    p_trainer.train(epochs, save_every = 10)

    dist.destroy_process_group()
    print(f"Process group {rank} destroyed; Training finished")


if "__main__" == __name__:

    parser = argparse.ArgumentParser(description="TimeSFormer")

    parser.add_argument("--epochs", type=int, default=100) # Check the code's feasibility
    parser.add_argument("--num_gpus", type=int, default=1) # Check multi-gpu
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--csv_file", type=str, default= "./datasets/updated_fold_csv_files/MDMER_fold_csv/MDMER_dataset_updated_fold0.csv")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--pretrained_dir", type=str, default="./pretrained")
    parser.add_argument('--pretrained',
                        action='store_true',
                        help='Choose pretrained backbone or scratch for swin and hicmae')
    parser.add_argument("--experiment_name", type=str, default="mdmer_vivit-test-eeg_5e-5_b12_e100_res224")
    parser.add_argument("--dataset", type=str, default="mdmer", choices=('eav', 'mdmer', 'emognition'))
    parser.add_argument("--model", type=str, default="vivit", choices=('vivit', 'swin', 'tsf', 'hicmae', 'tvlt'))
    parser.add_argument("--port", type=str, default="8008")
    parser.add_argument("--seed", type=int, default=7254)
    parser.add_argument("--img_size", type=int, default="224")

    parser.add_argument('--eeg_signal',
                        action='store_true',
                        help='Choose video or video+EEG by arg option')
    parser.add_argument('--fft_mode', type=str, default='Spectrogram',
                        choices=('AbsFFT', 'Spectrogram'), help='Choose FFT transformation method')
    parser.add_argument('--server', type=str, default='j_zay', choices=('j_zay', 'nef'), help='Choose which server for appropriate settings')

    args = parser.parse_args()

    # Add master information
    if('nef' == args.server):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = args.port

    # Spawning (<- run function) with each rank of gpu

        if args.num_gpus > 1:
            spawn_context = mp.spawn(
                run,
                nprocs=args.num_gpus,
                args=(args,), join=False
            )

            while not spawn_context.join():
                pass

            for process in spawn_context.processes:
                if process.is_alive():
                    process.terminate()
                process.join()
        else:
            run(0, args)
    else:
        run(-1, args)


    print("TRAINING IS DONE")
