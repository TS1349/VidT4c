import os
import sys
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
from models import BridgedTimeSFormer4C, BridgedViViT4C, BridgedVideoSwin4C, VEMT, \
    MilmerModel, EAVModel, \
    Medformer_Model, EEG_Transformer, PatchTST_Model, Crossformer_Model, Informer_Model, FEDformer_Model, \
    CBraMod_Model, LaBraM_Model, DGCNN_Model, GCBNet_Model, Biot_Model, ST_Model
from dataloader import EAVDataset, EmognitionDataset, MDMERDataset, dense_video_collate_fn
from dataloader.transforms import AbsFFT, STFTFixedSize, MedformerEEGPreprocess, CBraModTransform
# from scheduler import CosineScheduler

from torch.nn.parallel import DistributedDataParallel as DDP

import argparse
from collections import OrderedDict

import gc

class FocalLoss(torch.nn.Module):
    def __init__(self, weight=None, gamma=1.5, reduction="mean"):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, logits, target):
        ce = torch.nn.functional.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        fl = ((1 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return fl.mean()
        elif self.reduction == "sum":
            return fl.sum()
        return fl

def get_torch_model(name):
    # Video model
    if name == "vivit":
        return BridgedViViT4C
    elif name == "tsf":
        return BridgedTimeSFormer4C
    elif name == "swin":
        return BridgedVideoSwin4C
    # EEG model (https://github.com/DL4mHealth/Medformer/tree/main)
    elif name == "medformer":
        return Medformer_Model
    elif name == "eegtransformer":
        return EEG_Transformer
    elif name == "patchtst":
        return PatchTST_Model
    elif name == "crossformer":
        return Crossformer_Model
    elif name == "fedformer":
        return FEDformer_Model
    elif name == "informer":
        return Informer_Model
    # Ours (EEG-Mamba + Video)
    elif name == "vemt":
        return VEMT
    # V+EEG benchmark baselines (published, code-public)
    elif name == "milmer":
        return MilmerModel
    elif name == "eav":
        return EAVModel
    # EEG emotion recognition models
    elif name == "cbramod":
        return CBraMod_Model
    elif name == "labram":
        return LaBraM_Model
    elif name == "dgcnn":
        return DGCNN_Model
    elif name == "gcbnet":
        return GCBNet_Model
    elif name == "biot":
        return Biot_Model
    elif name == 'sttransformer':
        return ST_Model
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

def load_flexible(model, ckpt_path, *,
                  pick_keys=None,
                  drop_prefixes=(),
                  drop_startswith=(),
                  copy_bibranch=False,
                  allow_mismatch_prefixes=(),
                  init_bibranch_fallback=False,
                  verbose=True):
    ckpt = torch.load(ckpt_path, map_location='cpu')

    sd = None
    if isinstance(ckpt, dict):
        for k in (pick_keys or ['state_dict','model','net','weights']):
            if k in ckpt and isinstance(ckpt[k], dict):
                sd = ckpt[k]; break
        if sd is None:
            sd = ckpt
    else:
        sd = ckpt

    new_sd = OrderedDict()
    for k, v in sd.items():
        if any(k.startswith(pfx) for pfx in drop_startswith):
            continue
        for p in drop_prefixes:
            if k.startswith(p):
                k = k[len(p):]
        new_sd[k] = v

    msd = model.state_dict()
    def _allowed_mismatch(k):
        return any(k.startswith(p) for p in allow_mismatch_prefixes)

    filtered = {}
    for k, v in new_sd.items():
        if k in msd:
            if msd[k].shape == v.shape or _allowed_mismatch(k):
                filtered[k] = v

    if copy_bibranch:
        for k in msd.keys():
            if '_b.' in k and k not in filtered:
                k_base = k.replace('_b.', '.')
                if k_base in filtered and msd[k].shape == filtered[k_base].shape:
                    filtered[k] = filtered[k_base].clone()
                elif k_base in new_sd and msd[k].shape == new_sd[k_base].shape:
                    filtered[k] = new_sd[k_base].clone()

    missing, unexpected = model.load_state_dict(filtered, strict=False)

    if init_bibranch_fallback and missing:
        with torch.no_grad():
            for name in list(missing):
                if '_b.' in name and name in msd:
                    base = name.replace('_b.', '.')
                    t = msd[name]
                    if base in model.state_dict():
                        model.state_dict()[name].copy_(model.state_dict()[base])
                    else:
                        if t.ndim == 0:
                            t.copy_(torch.ones_like(t))
                        else:
                            t.copy_(torch.zeros_like(t))
        missing = [m for m in missing if not ('_b.' in m)]

    if verbose:
        print(f"[flex-load] {os.path.basename(ckpt_path)} -> loaded={len(filtered)} "
              f"missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("  missing(sample):", missing[:12], '...' if len(missing) > 12 else '')
        if unexpected:
            print("  unexpected(sample):", unexpected[:12], '...' if len(unexpected) > 12 else '')
    return missing, unexpected

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

def run(
    rank, args
):
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
    # experiment_name = args.experiment_name
    base_exp = args.experiment_name
    checkpoint_dir =args.checkpoint_dir
    torch_model = get_torch_model(args.model)
    fft_mode = args.fft_mode
    img_size = args.img_size

    torch_dataset = get_torch_dataset(args.dataset)

    print(f"dataset: {torch_dataset}")
    print(f"model: {torch_model}")

    # Distribute each GPUS by rank
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=args.num_gpus, rank=rank)

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    else:
        print("No GPU available")
 
    # Data processing
    video_preprocessor_train = Compose([
        Resize(size=(img_size, img_size), antialias=True),
        RandomHorizontalFlip(p=0.5),
        ColorJitter(brightness=0.2, contrast=0.2),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225)),
    ])

    video_preprocessor_val = Compose([
        Resize(size=(img_size, img_size), antialias=True),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225)),
    ])

    # Check the right fourier transformation
    fft_val = None

    if fft_mode == "AbsFFT":
        fft = AbsFFT(dim=-2)
        freq_bin = 64

    elif fft_mode == "Spectrogram":
        if args.model in ('vivit','swin','tsf'):
            freq = 224
            time = 224
        else:
            freq = 128
            time = 256

        if args.dataset == 'emognition':
            hop_length = 512
        else:
            hop_length = 128

        fft = STFTFixedSize(n_fft = hop_length, target_freq=freq, target_time=time)
        freq_bin = freq * time

    elif fft_mode == "Medraw": # Medformer style EEG processing
        if args.dataset == 'emognition':
            eeg_sf_in = 256
        elif args.dataset == 'mdmer':
            eeg_sf_in = 300
        elif args.dataset == 'eav':
            eeg_sf_in = 500
            
        fft = MedformerEEGPreprocess(
                sfreq_in=eeg_sf_in,
                sfreq_out=256,
                band=(0.5, 45.0),
                seg_len=256,
                seg_overlap=0.5
            )

    elif fft_mode == "cbramod":
        fft = CBraModTransform(target_len=2000, patch_size=200)
        fft_val = fft

        freq = 128
        time = 256

        if args.dataset == 'emognition':
            hop_length = 512
        else:
            hop_length = 128

        fft_local = STFTFixedSize(n_fft = hop_length, target_freq=freq, target_time=time)
        freq_bin = freq * time


    if fft_val is None:
        fft_val = fft

    _num_clips = getattr(args, 'num_clips', 1)
    _frame_interval = getattr(args, 'frame_interval', 2)
    _eeg_full_signal = getattr(args, 'eeg_full_signal', False)
    _fps_normalize = getattr(args, 'fps_normalize', False)
    _dense_video_clips = getattr(args, 'dense_video_clips', False)
    _dense_cache_features = getattr(args, 'dense_cache_features', False)
    _clip_overlap_ratio = float(getattr(args, 'clip_overlap_ratio', 0.0))

    # Dense feature cache needs train == val preprocessing (no random flip/jitter)
    # so cached intermediates stay valid across epochs.
    if _dense_cache_features:
        if rank == 0:
            print('[dense_cache_features] training augmentation disabled '
                  '(train_transform = val_transform) so cached features stay consistent.')
        video_preprocessor_train = video_preprocessor_val
    training_dataset = torch_dataset(
        csv_file=csv_file,
        time_window = 10.0, #sec -> change to avoid duplication of eeg sampling
        video_transform=video_preprocessor_train,
        eeg_transform = fft,
        eeg_transform_local=fft_local,
        split = "train",
        num_clips=_num_clips,
        frame_interval=_frame_interval,
        eeg_full_signal=_eeg_full_signal,
        fps_normalize=_fps_normalize,
        dense_video_clips=_dense_video_clips,
        clip_overlap_ratio=_clip_overlap_ratio,
    )
    validation_dataset = torch_dataset(
        csv_file=csv_file,
        time_window = 10.0, #sec
        video_transform=video_preprocessor_val,
        eeg_transform = fft_val,
        eeg_transform_local=fft_local,
        split = "test",
        num_clips=_num_clips,
        frame_interval=_frame_interval,
        eeg_full_signal=_eeg_full_signal,
        fps_normalize=_fps_normalize,
        dense_video_clips=_dense_video_clips,
        clip_overlap_ratio=_clip_overlap_ratio,
    )

    # Eval-only dump: emit stable per-sample id (index/path) for cross-model alignment.
    if getattr(args, 'dump_predictions', False):
        validation_dataset.return_meta = True

    training_sampler = DistributedSampler(
        dataset=training_dataset,
        num_replicas=args.num_gpus,
        rank=rank,
    )

    # Add validation sampler
    validation_sampler = DistributedSampler(
        dataset=validation_dataset,
        num_replicas=args.num_gpus,
        rank=rank,
        shuffle=False
    )

    # Custom collate for dense-video mode (variable N clips per sample).
    _collate_fn = dense_video_collate_fn if _dense_video_clips else None
    # Dense mode: each sample's video tensor is ~1 GB. 4 workers × prefetch 2
    # would put ~32 GB into /dev/shm + CPU RAM and crash with SIGKILL during the
    # slow first epoch (cache miss path). Trim worker count and prefetch buffer
    # in dense mode; normal mode keeps the original throughput-optimized values.
    if _dense_video_clips:
        _num_workers = 2
        _prefetch = 1
    else:
        _num_workers = 4
        _prefetch = 2

    # We don't need shuffle when using DistributedSampler
    training_dataloader = DataLoader(
        dataset=training_dataset,
        batch_size=batch_size,
        num_workers=_num_workers,
        prefetch_factor=_prefetch if _num_workers > 0 else None,
        pin_memory=False,
        persistent_workers=False,
        sampler=training_sampler, drop_last=False,
        collate_fn=_collate_fn,
        # multiprocessing_context="spawn"
    )

    # Val-dataloader shouldn't be sampled by training_sampler
    validation_dataloader = DataLoader(
        dataset=validation_dataset,
        num_workers=_num_workers,
        prefetch_factor=_prefetch if _num_workers > 0 else None,
        batch_size=batch_size,
        pin_memory=False,
        persistent_workers=False,
        sampler=validation_sampler, drop_last=False,
        collate_fn=_collate_fn,
        # multiprocessing_context="spawn"
    )


    if args.model =='hicmae':
        output_dim = training_dataset.output_shape
        model = torch_model(args, output_dims=output_dim, all_frames=32, eeg_channels = training_dataset.eeg_channel_count) # Should get num_class from the dataloader
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

            orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens)//(num_frames // model.encoder.patch_embed.tubelet_size)) ** 0.5)
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
            freq_size = model.encoder_audio.patch_embed.patch_hw[1] # assert the freq dim is fixed (i.e., 128//16=8)
            orig_temporal_size = (pos_embed_checkpoint.shape[-2] - num_extra_tokens) // freq_size
            new_temporal_size = model.encoder_audio.patch_embed.patch_hw[0]
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

    elif args.model in ('milmer', 'eav'):
        # V+EEG baselines; same constructor as VEMT, HF backbones self-load.
        model = torch_model(args,
                    output_dim = training_dataset.output_shape,
                    image_size = img_size,
                    eeg_channels = training_dataset.eeg_channel_count,
                    frequency_bins = freq_bin,
        )

    elif args.model == 'vemt':
        model = torch_model(args,
                    output_dim = training_dataset.output_shape,
                    image_size = img_size,
                    eeg_channels = training_dataset.eeg_channel_count,
                    frequency_bins = freq_bin,
        )

        # ViViT/TSF/Swin self-load pretrained in their wrappers; only
        # AdaMAE/VideoMAE go through load_flexible below.
        if (args.pretrained and not args.set_eeg_only
                and getattr(args, 'vemt_video', 'AdaMAE') not in ('ViViT', 'TSF', 'Swin')):
            try:
                if args.vemt_video == "AdaMAE":
                    pretrained_path = os.path.join(os.getcwd(), 'pretrained', 'checkpoint-199.pth') # AdaMAE
                else:
                    pretrained_path = os.path.join(os.getcwd(), 'pretrained', args.model + '_v0.pth')  # VideoMAE
            except AttributeError:
                pretrained_path = os.path.join(os.getcwd(), 'pretrained', args.model + '_v0.pth')  # VideoMAE

            print(f'Loading pretrained weights from {pretrained_path}')
            load_flexible(
                model.video_model,
                pretrained_path,
                pick_keys=['model'],
                drop_prefixes=('encoder.',),
                drop_startswith=('head.', 'fc_norm.', 'fc.'),
                copy_bibranch=False,
                allow_mismatch_prefixes=('patch_embed', 'pos_embed'),
                verbose=True
            )

        if args.pretrained and not args.set_video_only:
            # EEG backbone pretrained loading. CBraMod/LaBraM load from disk;
            # REVE self-loads from HF inside its __init__ (no-op here).
            if getattr(args, 'eeg_backbone', 'cbramod') == 'reve':
                print('[REVE] EEG backbone self-loads from HuggingFace inside REVE_Model.__init__ — skipping cbramod.pth.')
            elif getattr(args, 'eeg_backbone', 'cbramod') == 'labram':
                pretrained_path = os.path.join(os.getcwd(), 'pretrained', 'labram.pth')  # LaBraM
                print(f'Loading pretrained weights from {pretrained_path} (into eeg_model.labram)')
                load_flexible(model.eeg_model.labram, pretrained_path, drop_prefixes=('student.',))
            else:
                pretrained_path = os.path.join(os.getcwd(), 'pretrained', 'cbramod.pth')  # CBraMod
                print(f'Loading pretrained weights from {pretrained_path}')
                checkpoint = torch.load(pretrained_path, map_location='cpu')
                _load_state_dict(model.eeg_model, checkpoint)

        _kclip_active = getattr(args, 'num_clips', 1) > 1
        _dense_active = getattr(args, 'dense_video_clips', False)
        # Auto-freeze backbones under K-clip / dense multi-clip training.
        if _kclip_active or _dense_active:
            n_unfreeze = int(getattr(args, 'video_unfreeze_last_n_blocks', 0))
            n_unfreeze_e = int(getattr(args, 'eeg_unfreeze_last_n_blocks', 0))
            eeg_full_unfreeze = bool(getattr(args, 'eeg_full_unfreeze', False))
            model.freeze_backbones(
                video_unfreeze_last_n=n_unfreeze,
                eeg_unfreeze_last_n=n_unfreeze_e,
                eeg_full_unfreeze=eeg_full_unfreeze,
            )
            if rank == 0:
                _v_msg = (
                    f"video_model last {n_unfreeze} block(s) + norm/head trainable"
                    if n_unfreeze > 0 else "video_model head only"
                )
                if eeg_full_unfreeze:
                    _e_msg = "CBraMod fully unfrozen (encoder + patch_embedding + classifier)"
                elif n_unfreeze_e > 0:
                    _e_msg = f"CBraMod last {n_unfreeze_e} layer(s) + classifier trainable"
                else:
                    _e_msg = "CBraMod classifier only"
                _mode_tag = (
                    f"[dense N-clip] fi={args.frame_interval}"
                    if _dense_active else f"[K-clip] num_clips={args.num_clips}"
                )
                print(
                    f"{_mode_tag}: backbones frozen — "
                    f"{_v_msg}; {_e_msg}."
                )

        # On-disk dense feature cache; dir auto-derived from dataset/csv/backbone/fi/
        # unfreeze split unless --dense_cache_dir is set.
        if _dense_cache_features and hasattr(model, 'setup_dense_cache'):
            _explicit_dir = getattr(args, 'dense_cache_dir', '') or ''
            if _explicit_dir:
                _cache_dir = _explicit_dir
            else:
                _csv_base = os.path.splitext(os.path.basename(args.csv_file))[0]
                _backbone = getattr(args, 'vemt_video', 'unknown')
                _n_unfreeze = int(getattr(args, 'video_unfreeze_last_n_blocks', 0))
                _fps_norm_tag = '_fpsnorm' if getattr(args, 'fps_normalize', False) else ''
                _ov = float(getattr(args, 'clip_overlap_ratio', 0.0))
                _overlap_tag = f'_ov{int(round(_ov * 100))}' if _ov > 0 else ''
                _cache_dir = os.path.join(
                    '.feature_cache',
                    args.dataset,
                    _csv_base,
                    _backbone,
                    f'fi{args.frame_interval}{_fps_norm_tag}{_overlap_tag}_unfreeze{_n_unfreeze}',
                )
            model.setup_dense_cache(_cache_dir)
            if rank == 0:
                print(f'[dense_cache_features] disk cache dir: {_cache_dir}')


    elif args.model in ('medformer', 'eegtransformer', 'patchtst', 'crossformer', 'fedformer', 'informer'):
        model = torch_model(args, output_dim=training_dataset.output_shape, enc_in=training_dataset.eeg_channel_count, seq_len=256)

    # elif args.model in ('cbramod', 'dgcnn', 'gcbnet'):
    elif args.model in ('cbramod', 'labram', 'biot', 'sttransformer'):
        model = torch_model(args, output_dim=training_dataset.output_shape, in_chans=training_dataset.eeg_channel_count)

        if args.model == 'cbramod':
            pretrained_path = os.path.join(os.getcwd(), 'pretrained', args.model + '.pth')
            print(f'Loading pretrained weights from {pretrained_path}')
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            _load_state_dict(model, checkpoint)  # Backbone is defined as the model.model


        elif args.model == 'labram':
            pretrained_path = os.path.join(os.getcwd(), 'pretrained', args.model + '.pth')
            print(f'Loading pretrained weights from {pretrained_path}')
            load_flexible(
                model,
                pretrained_path,
                drop_prefixes=('student.',),
            )

        elif args.model == 'biot': # Should change to fit
            pretrained_path = os.path.join(os.getcwd(), 'pretrained', args.model + '.ckpt')
            print(f'Loading pretrained weights from {pretrained_path}')
            load_flexible(
                model.biot,
                pretrained_path,
            )

    else:
        model = torch_model(args,
                    output_dim = training_dataset.output_shape,
                    image_size = img_size,
                    eeg_channels = training_dataset.eeg_channel_count,
                    frequency_bins = freq_bin,
        )
        
        if args.pretrained and args.model == 'tsf': # ViViT, Swin utilize inner weight
            pretrained_path = os.path.join(os.getcwd(), 'pretrained', args.model + '.pth') # TimeSFormer
            print(f'Loading pretrained weights from {pretrained_path}')
            checkpoint = torch.load(pretrained_path)
            model.model.load_state_dict(checkpoint, strict=False) # Backbone is defined as the model.model

    # # Check sample distribution via preateind weight
    # checkpoint = torch.load(args.pretrained_dir)
    # model.load_state_dict(checkpoint, strict=False)

    device = torch.device(f"cuda:{rank}")
    model.to(device)

    # --relresfuse reuses backbone features across aux branches → needs
    # static_graph=True so DDP's reducer doesn't trip on reused params.
    _static_graph = bool(getattr(args, 'relresfuse', False))
    model = DDP(model, device_ids=[rank],
            find_unused_parameters=(not _static_graph),
            static_graph=_static_graph)

    stats = getattr(training_dataset, "class_stats", None)
    val_w, aro_w = None, None

    if stats is not None:
        val_ratios = stats["valence_ratios"]
        aro_ratios = stats["arousal_ratios"]

        val_w = torch.tensor([val_ratios[i] for i in range(5)], dtype=torch.float32, device=device)
        aro_w = torch.tensor([aro_ratios[i] for i in range(5)], dtype=torch.float32, device=device)

        val_w = 1.0 / (val_w + 1e-12)
        aro_w = 1.0 / (aro_w + 1e-12)

        val_w = val_w / val_w.mean()
        aro_w = aro_w / aro_w.mean()

    # Using balanced CE loss + Focal loss for training
    loss_function = FocalLoss(gamma=1.5).to(device)

    label_smoothing = 0.05
    ce_val = torch.nn.CrossEntropyLoss(weight=val_w, label_smoothing=label_smoothing).to(device)
    ce_aro = torch.nn.CrossEntropyLoss(weight=aro_w, label_smoothing=label_smoothing).to(device)

    def build_param_groups(model, args):
        base_lr = args.learning_rate
        head_lr = getattr(args, "head_learning_rate", None)
        gcn_lr = getattr(args, "gcn_learning_rate", None)
        # Optional separate LR for the EEG backbone; None keeps it at base_lr.
        eeg_bb_lr = getattr(args, "eeg_backbone_learning_rate", None)

        if head_lr is None:
            head_lr = base_lr
        if gcn_lr is None:
            gcn_lr = base_lr * 5.0

        backbone_wd = args.weight_decay
        head_wd = getattr(args, "head_weight_decay", 0.01)
        gcn_wd = getattr(args, "gcn_weight_decay", 0.01)

        no_decay_keywords = ["bias", "norm", "bn", "ln", "layernorm"]

        groups = {
            "backbone_decay": {
                "params": [], "lr": base_lr, "weight_decay": backbone_wd, "name": "backbone_decay"
            },
            "backbone_no_decay": {
                "params": [], "lr": base_lr, "weight_decay": 0.0, "name": "backbone_no_decay"
            },
            "head_decay": {
                "params": [], "lr": head_lr, "weight_decay": head_wd, "name": "head_decay"
            },
            "head_no_decay": {
                "params": [], "lr": head_lr, "weight_decay": 0.0, "name": "head_no_decay"
            },
            "gcn_decay": {
                "params": [], "lr": gcn_lr, "weight_decay": gcn_wd, "name": "gcn_decay"
            },
            "gcn_no_decay": {
                "params": [], "lr": gcn_lr, "weight_decay": 0.0, "name": "gcn_no_decay"
            },
        }
        if eeg_bb_lr is not None:
            groups["eeg_decay"] = {
                "params": [], "lr": eeg_bb_lr, "weight_decay": backbone_wd, "name": "eeg_decay"
            }
            groups["eeg_no_decay"] = {
                "params": [], "lr": eeg_bb_lr, "weight_decay": 0.0, "name": "eeg_no_decay"
            }

        head_keywords = [
            "video_model.head",
            "eeg_model.classifier",
        ]
        # EEG backbone (everything under eeg_model EXCEPT its classifier, which is
        # caught by head_keywords above). Only routed to its own group when
        # --eeg_backbone_learning_rate is set.
        eeg_keywords = ["eeg_model"]

        gcn_keywords = [
            "gcn_local",
            "gcn_region",
            "region_pool",
            "eeg_feat_proj",
        ]

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue

            is_no_decay = (
                    p.ndim <= 1
                    or any(k in name.lower() for k in no_decay_keywords)
            )

            if any(k in name for k in gcn_keywords):
                group_key = "gcn_no_decay" if is_no_decay else "gcn_decay"
            elif any(k in name for k in head_keywords):
                group_key = "head_no_decay" if is_no_decay else "head_decay"
            elif eeg_bb_lr is not None and any(k in name for k in eeg_keywords):
                group_key = "eeg_no_decay" if is_no_decay else "eeg_decay"
            else:
                group_key = "backbone_no_decay" if is_no_decay else "backbone_decay"

            groups[group_key]["params"].append(p)

        final_groups = [g for g in groups.values() if len(g["params"]) > 0]

        if not torch.distributed.is_available() or not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print("\n[Optimizer param groups]")
            for g in final_groups:
                n_params = sum(p.numel() for p in g["params"])
                print(
                    f"  {g['name']}: "
                    f"params={n_params:,}, lr={g['lr']}, wd={g['weight_decay']}"
                )
            print()

        return final_groups

    param_groups = build_param_groups(model, args)

    optimizer = optim.AdamW(
        params=param_groups,
        eps=1e-8,
        betas=(0.9, 0.999),
    )

    # Scheduler: delay decay until after expert warmup

    steps_per_epoch = len(training_dataloader)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(0.1 * total_steps)

    scheduler_start_epoch = args.lrscheduler_start

    milestone_epochs = list(range(scheduler_start_epoch, 1000, args.lrscheduler_step))
    milestone_steps = [m * steps_per_epoch for m in milestone_epochs]

    decay = args.lrscheduler_decay

    def multistep_with_warmup_lambda(global_step: int):
        if global_step < warmup_steps:
            return float(global_step + 1) / float(max(1, warmup_steps))

        k = 0
        for ms in milestone_steps:
            if global_step >= ms:
                k += 1
            else:
                break

        return decay ** k

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=multistep_with_warmup_lambda
    )

    p_trainer = PTrainer(
        model=model,
        lr_scheduler = lr_scheduler,
        training_dataloader=training_dataloader,
        validation_dataloader=validation_dataloader,
        optimizer=optimizer,
        loss_function=loss_function,
        loss_function_v=ce_val,
        loss_function_a=ce_aro,
        fusion = args.fusion,
        weight_ce=args.w_ce,
        checkpoint_dir=checkpoint_dir,
        gpu=device,
        gpu_id=rank,
        experiment_name=args.experiment_name,
        patience=args.patience,
        per_clip_aux_loss=getattr(args, 'per_clip_aux_loss', 0.0),
        grad_clip=getattr(args, 'grad_clip', 0.0),
    )

    # Eval-only dump: load best.pt, write per-sample predictions. Run --num_gpus 1.
    if getattr(args, 'dump_predictions', False):
        ckpt_path = args.dump_ckpt
        sd = torch.load(ckpt_path, map_location=device)
        if isinstance(sd, dict) and 'model' in sd and all(isinstance(k, str) for k in sd):
            sd = sd.get('model', sd)
        inner = p_trainer.model.module if hasattr(p_trainer.model, 'module') else p_trainer.model
        missing, unexpected = inner.load_state_dict(sd, strict=False)
        if rank == 0:
            print(f'[dump] loaded {ckpt_path} -> missing={len(missing)} unexpected={len(unexpected)}')
        p_trainer.dump_affinity = getattr(args, 'dump_affinity', False)
        p_trainer.test_only(save_dir=args.dump_dir, model_name=args.dump_name)
        dist.destroy_process_group()
        torch.cuda.empty_cache()
        gc.collect()
        # Exit before multi_run_main's metric aggregation (jsonl already flushed).
        sys.stdout.flush()
        os._exit(0)

    best_metrics = p_trainer.train(epochs, save_every = 10)

    dist.destroy_process_group()
    torch.cuda.empty_cache()
    gc.collect()

    return best_metrics


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def multi_run_main(args):
    num_runs = int(getattr(args, 'num_runs', 1))
    results = []

    base_exp = args.experiment_name
    base_seed = args.seed
    start_run = int(getattr(args, 'start_run', 0))

    # Load already-completed runs from best.txt
    for run_id in range(start_run):
        best_path = os.path.join(args.checkpoint_dir, base_exp, f"run_{run_id}", "best.txt")
        with open(best_path, "r") as f:
            last_line = f.readlines()[-1]
        metrics = {}
        for item in last_line.strip().split(","):
            k, v = item.split("=")
            metrics[k] = float(v)
        results.append(metrics)
        print(f"[start_run] Loaded run_{run_id} from best.txt: {metrics}")

    for run_id in range(start_run, num_runs):
        print(f"\n================ RUN {run_id} ================\n")

        # Seed change
        args.experiment_name = f"{base_exp}/run_{run_id}"
        args.seed = base_seed + run_id

        set_seed(args.seed)

        # Using spawn same with original code
        if args.num_gpus > 1:
            spawn_context = mp.spawn(
                run,
                nprocs=args.num_gpus,
                args=(args,),
                join=False
            )

            while not spawn_context.join():
                pass

            for process in spawn_context.processes:
                if process.is_alive():
                    process.terminate()
                process.join()

            # get from the best.txt
            best_path = os.path.join(args.checkpoint_dir, base_exp, f"run_{run_id}", "best.txt")

            with open(best_path, "r") as f:
                last_line = f.readlines()[-1]

            metrics = {}
            for item in last_line.strip().split(","):
                k, v = item.split("=")
                metrics[k] = float(v)

            results.append(metrics)

        else:
            metrics = run(0, args)   # single GPU
            results.append(metrics)

    # Statistic calculation
    accs = [r["acc"] for r in results]
    uars = [r["uar"] for r in results]
    f1m  = [r["f1_macro"] for r in results]
    f1w  = [r["f1_weighted"] for r in results]

    print("\n================ FINAL RESULT ================\n")
    print(f"ACC : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"UAR : {np.mean(uars):.4f} ± {np.std(uars):.4f}")
    print(f"F1m : {np.mean(f1m):.4f} ± {np.std(f1m):.4f}")
    print(f"F1w : {np.mean(f1w):.4f} ± {np.std(f1w):.4f}")

    save_path = os.path.join(args.checkpoint_dir, args.experiment_name, "final_results.txt")
    with open(save_path, "w") as f:
        f.write("===== FINAL RESULT =====\n")
        f.write(f"ACC : {np.mean(accs):.4f} ± {np.std(accs):.4f}\n")
        f.write(f"UAR : {np.mean(uars):.4f} ± {np.std(uars):.4f}\n")
        f.write(f"F1m : {np.mean(f1m):.4f} ± {np.std(f1m):.4f}\n")
        f.write(f"F1w : {np.mean(f1w):.4f} ± {np.std(f1w):.4f}\n")

    print(f"\nSaved final results to {save_path}")

if "__main__" == __name__:

    parser = argparse.ArgumentParser(description="TimeSFormer")

    parser.add_argument("--epochs", type=int, default=50) # Check the code's feasibility
    parser.add_argument("--num_gpus", type=int, default=1) # Check multi-gpu
    parser.add_argument("--batch_size", type=int, default=2)

    parser.add_argument("--lrscheduler_start", type=int, default=10) # default: 10
    parser.add_argument("--lrscheduler_step", type=int, default=10)
    parser.add_argument("--lrscheduler_decay", type=float, default=0.75) 

    parser.add_argument("--learning_rate", type=float, default=1e-4) # 2e-5: transformer, 1e-3: mamba, foundation: 1e-4
    parser.add_argument("--weight_decay", type=float, default=0.05) # 2e-2: transformer, foundation, mamba: 5e-2
    parser.add_argument("--csv_file", type=str, default= "./datasets/updated_fold_csv_files/Emognition_fold_csv/Emognition_dataset_updated_fold0.csv")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_5")
    # parser.add_argument("--pretrained_dir", type=str, default="./pretrained/mdmer_cbramod_run1.pt")
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Choose pretrained backbone or scratch for swin and hicmae')
    # parser.add_argument("--experiment_name", type=str, default="mdmer-vemt-va5_gcn-two_mile5-0.75_1e-5-5e-5-m0.1_b4_e50_res224")
    parser.add_argument("--experiment_name", type=str, default="emognition_check_1e-4-5e-2_b16_e50")
    parser.add_argument("--dataset", type=str, default="emognition", choices=('eav', 'mdmer', 'emognition'))
    parser.add_argument("--model", type=str, default="vemt", choices=('vivit', 'swin', 'tsf', 'vemt',
                                                                           'milmer', 'eav',
                                                                           'medformer', 'eegtransformer', 'patchtst', 'crossformer', 'fedformer', 'informer',
                                                                             'cbramod', 'labram', 'dgcnn', 'gcbnet', "biot", "sttransformer" ))
    parser.add_argument("--port", type=str, default="20025")
    parser.add_argument("--seed", type=int, default=7255)
    parser.add_argument("--start_run", type=int, default=0, help="Resume multi-run from this run_id (previous runs loaded from best.txt)")
    parser.add_argument("--num_runs", type=int, default=1, help="Total number of multi-seed runs (default 1 = single seed). Set 3 or 5 for repeated experiments.")
    parser.add_argument("--img_size", type=int, default="224")

    parser.add_argument('--eeg_signal', action='store_true', default=False,
                        help='Choose video+EEG input or model')
    parser.add_argument('--set_eeg_only', action='store_true', default=False,
                        help='Using only eeg in VEMT')
    parser.add_argument('--set_video_only', action='store_true', default=False,
                        help='Using only video in VEMT')
    parser.add_argument('--fft_mode', type=str, default='cbramod',
                        choices=('AbsFFT', 'Spectrogram', 'Medraw', 'cbramod'), help='Choose FFT transformation method')
    parser.add_argument('--gcn', action='store_true', default=False,
                    help='Using gcn as classifier')

    parser.add_argument("--w_ce", type=float, default=0.9)

    parser.add_argument('--fusion', type=str, default='naive',
                        choices=('FiLM', 'attention', 'naive', 'gated_mixture',
                                 'region'), help='Choose fusion method')

    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping patience (epochs without improvement). 0 to disable.')

    parser.add_argument('--vemt_video', type=str, default='AdaMAE',
                        choices=('VideoMAE', 'AdaMAE', 'ViViT', 'TSF', 'Swin'),
                        help='Video backbone inside VEMT. ViViT/TSF/Swin load their own pretrained '
                             'in __init__; AdaMAE/VideoMAE go through load_flexible in runner.py.')
    parser.add_argument('--eeg_backbone', type=str, default='cbramod',
                        choices=('cbramod', 'reve', 'labram'),
                        help='EEG backbone inside VEMT. cbramod (default) = legacy CBraMod path '
                             '(loads pretrained/cbramod.pth). reve = REVE-EEG (NeurIPS 2025, '
                             'HF brain-bzh/reve-base, gated) — requires `huggingface-cli login` + '
                             'RUA acceptance. Both share the same downstream contract '
                             '(GCN region / fusion / classifier untouched).')
    parser.add_argument('--num_clips', type=int, default=1,
                        help='K-clip multi-clip inference: K representative clips at backbone-native '
                             'temporal stride. 1 = original single stretched clip.')
    parser.add_argument('--video_unfreeze_last_n_blocks', type=int, default=0,
                        help='Under K-clip auto-freeze (num_clips>1), keep the last N transformer '
                             'blocks of video_model trainable along with norm/fc_norm/head. '
                             '0 = full freeze (head only). e.g. 2 = last 2 blocks fine-tuned.')
    parser.add_argument('--eeg_unfreeze_last_n_blocks', type=int, default=0,
                        help='Under K-clip auto-freeze, keep the last N CBraMod encoder layers '
                             '(self.eeg_model.encoder.layers[-N:]) trainable along with classifier. '
                             '0 = classifier-only (linear probing). Use to symmetrically fine-tune '
                             'EEG and video backbones (e.g. both at 2 blocks).')
    parser.add_argument('--eeg_full_unfreeze', action='store_true', default=False,
                        help='Under K-clip auto-freeze, fully unfreeze CBraMod (encoder + '
                             'patch_embedding + classifier). Overrides --eeg_unfreeze_last_n_blocks. '
                             'CBraMod is small (~22.5M) so full FT is feasible even under K-clip. '
                             'Use to asymmetrically fully fine-tune EEG while video keeps partial freeze.')
    parser.add_argument('--per_clip_aux_loss', type=float, default=0.0,
                        help='Weight for per-clip auxiliary CE loss in K-clip set_video_only mode. '
                             'Each of K clips gets the video-level label and contributes a CE; the '
                             'mean is added to the pooled loss with this weight. 0 = disabled.')
    parser.add_argument('--grad_clip', type=float, default=0.0,
                        help='Max L2 norm for gradient clipping (torch.nn.utils.clip_grad_norm_). '
                             '0 = disabled (default, preserves prior behavior). 1.0 = standard. '
                             'Use when train_loss diverges mid-training (single-batch gradient '
                             'spikes destabilizing Adam moments) — observed on MDMER fold1 GCN.')
    parser.add_argument('--frame_interval', type=int, default=2,
                        help='Native frame stride per output step in K-clip mode. '
                             'Each clip covers num_out_frames * frame_interval native frames '
                             '(e.g. 32*2/30fps ≈ 2.13s). VideoMAE/AdaMAE typical: 2.')
    parser.add_argument('--fps_normalize', action='store_true', default=False,
                        help='Scale frame_interval by fps/30 so 30/60fps videos cover the same '
                             'wall-clock per clip. Use for mixed-fps datasets (e.g. Emognition: '
                             'fi=12 stays 12.8s on 30fps and becomes stride-24/12.8s on 60fps).')
    parser.add_argument('--dense_video_clips', action='store_true', default=False,
                        help='Dense N-clip video sampling: emit variable N non-overlapping clips '
                             'at native --frame_interval covering the WHOLE video (no temporal '
                             'sub-sampling, no gaps). N varies per sample; custom collate pads '
                             'to batch-max via cycle-repeat and adds video_lengths for mask-pool. '
                             'Model side processes clips in chunks (--dense_chunk_size, default 6) '
                             'and mean-pools valid clips. EEG forced to full-signal single-window '
                             'view. Designed for video backbones that prefer their native '
                             'short-clip distribution while still covering the full video. '
                             'Overrides --num_clips semantics for video.')
    parser.add_argument('--dense_chunk_size', type=int, default=6,
                        help='Chunk size (number of clips per inner forward) under --dense_video_clips. '
                             'Smaller = lower peak memory, more chunks = slightly slower per step.')
    parser.add_argument('--clip_overlap_ratio', type=float, default=0.0,
                        help='Under --dense_video_clips, fraction of overlap between consecutive '
                             'clips (0.0 = no overlap; 0.5 = 50% overlap → ~2× more clips with '
                             '50% temporal redundancy). Example: --frame_interval 4 (clip ≈ 4.3s '
                             '@ 30fps) + --clip_overlap_ratio 0.5 → consecutive clips overlap by '
                             '~2.1s. Cache dir auto-includes overlap tag.')
    parser.add_argument('--dense_cache_features', action='store_true', default=False,
                        help='Under --dense_video_clips, cache the output of the frozen part of the '
                             'video backbone (input to the trainable last N blocks) per sample. '
                             'Cache is shared across DDP ranks AND epochs via disk (see '
                             '--dense_cache_dir). After epoch 1 every sample is a cache hit → only '
                             'the trainable tail (blocks[-N:] + norm + head) re-runs each step. '
                             '~10x speedup. Forces train transform to match val (no random flip / '
                             'color jitter) so cached features stay valid across epochs. Requires '
                             'the backbone to expose forward_features_until/from (AdaMAE/modeling_finetune_v0).')
    parser.add_argument('--dense_cache_dir', type=str, default='',
                        help='Directory for the on-disk feature cache used with --dense_cache_features. '
                             'Empty → auto-derive from dataset+csv+frame_interval (e.g. '
                             '"./.feature_cache/<dataset>/<csv_basename>/fi<frame_interval>"). '
                             'All DDP ranks share this directory; first epoch populates, later '
                             'epochs and reruns of the same data setup get cache hits. Delete the '
                             'directory to force a fresh extraction (e.g. after changing backbone '
                             'checkpoint or video transform).')
    parser.add_argument('--clip_pool', type=str, default='mean',
                        choices=('mean', 'max', 'attn'),
                        help='Pooling strategy across K clips: mean (avg), max (element-wise max), '
                             'attn (learnable per-clip attention scores → weighted sum).')
    parser.add_argument('--eeg_full_signal', action='store_true', default=False,
                        help='When num_clips>1, do NOT split EEG into K clip-sized windows; '
                             'sample one full-view EEG window (K=1 behaviour) while video uses K clips.')
    parser.add_argument('--gcn_video_per_clip', action='store_true', default=False,
                        help='When num_clips>1 with --eeg_signal --gcn, expose each video clip as '
                             'a separate node in the GCN region graph (instead of pooling to one '
                             'video node). Lets the graph learn clip-clip / clip-region / clip-EEG '
                             'interactions. Auto-degenerates to single node when num_clips==1.')
    parser.add_argument('--gcn_temporal_adj', action='store_true', default=False,
                        help='Apply learnable Gaussian temporal-distance prior to GCN region '
                             'adjacency. Clip-clip edges between temporally distant clips are '
                             'softened by exp(-|t_i-t_j|^2 / 2σ^2), σ learnable (init=3 clips). '
                             'Region nodes are unaffected. Mitigates per-clip GCN over-smoothing '
                             'while keeping cross-modal pairing intact.')
    parser.add_argument('--gcn_clip_pe', action='store_true', default=False,
                        help='Add sinusoidal positional embedding to video/EEG clip nodes before '
                             'they enter the GCN region graph. Parameter-free, lets the GCN '
                             'distinguish "which clip in time" each node represents.')
    parser.add_argument('--adaptive_gate', action='store_true', default=False,
                        help='v2 fusion gate: replace the static, input-independent gate '
                             'self.weight (out=(1-sigmoid(w))*eeg + sigmoid(w)*video) with a '
                             'SAMPLE-adaptive MLP gate w=sigmoid(MLP([video_logit; eeg_logit])). '
                             'The static gate freezes at init (video share 0.27) because its '
                             'gradient cancels across samples that need video vs EEG; the '
                             'adaptive gate routes per-sample so its gradient no longer cancels. '
                             'Last layer init (zero weight, -1.0 bias) → identical to the static '
                             'baseline at step 0, then learns. arg-gated, default-off: baseline '
                             'untouched. Per-epoch mean video share logged to gate_log.txt.')
    parser.add_argument('--gcn_region_eeg_only', action='store_true', default=False,
                        help='Ablation A: cut video<->region edges in the GCN region graph. '
                             'With K_v dense video clips (e.g. 149), every region connects to '
                             'all clips → the shared video flood homogenizes regions (affinity: '
                             'per-region degree near-identical). This connects regions only to '
                             'EEG + other regions so they can differentiate. arg-gated, default-off.')
    parser.add_argument('--relresfuse', action='store_true', default=False,
                        help='Reliability-Guided Residual Fusion (distinct from the old '
                             '--reliability_gate / --rel_gate_v2 relgate flags). Unimodal aux '
                             'heads (kept CE-discriminative → de-corrupt shared backbone) + a '
                             'per-sample per-axis gate α supervised toward a live soft-CE '
                             'reliability target q=softmax(-CE/τ); output = GCN_fusion + '
                             'res·Σ(α_v·logit_v + α_e·logit_e) on DETACHED unimodal logits '
                             '(POST-GCN residual, so routing survives and GCN stays the floor). '
                             'arg-gated, default-off.')
    parser.add_argument('--relresfuse_res', type=float, default=0.0,
                        help='--relresfuse INITIAL value of the LEARNABLE residual scale. '
                             'Default 0.0 → output == baseline GCN at step 0; the residual '
                             'grows only where it helps (avoids perturbing a good baseline).')
    parser.add_argument('--relresfuse_kl', type=float, default=0.3,
                        help='--relresfuse weight on the gate KL(α || q) supervision loss.')
    parser.add_argument('--relresfuse_uni', type=float, default=0.5,
                        help='--relresfuse weight on the unimodal CE (keeps backbone discriminative).')
    parser.add_argument('--relresfuse_tau', type=float, default=0.5,
                        help='--relresfuse temperature for the soft-CE reliability target q.')
    parser.add_argument('--gcn_region_eeg_source', type=str, default='stft',
                        choices=('stft', 'cbramod'),
                        help='Per-channel EEG feature source for the GCN region nodes. '
                             '"stft" (default, legacy): hand-crafted STFT flattened via 25M Linear. '
                             '"cbramod": pre-positional CBraMod patch_embedding output (conv-on-raw '
                             '+ FFT projection, BEFORE the (19,7) positional encoding conv that mixes '
                             'channels). Per-channel independent, pretrained, 154K projection params. '
                             'Recommended for cleaner ablation and stronger features. '
                             'NOTE: with --eeg_backbone reve, "cbramod" transparently routes to REVE\'s '
                             'return_per_channel_pre (post-encoder, since REVE has no documented pre-pos '
                             'split). Same [B, ch, T_seg=10, 200] shape — downstream code unchanged.')
    parser.add_argument('--gcn_region_eeg_no_detach', action='store_true', default=False,
                        help='With --gcn_region_eeg_source cbramod, by default the per-channel CBraMod '
                             'features feeding the GCN region path are detached so the patch_embedding '
                             'is shaped only by EEG classification loss (stable, no objective conflict). '
                             'Pass this flag to disable detach and let GCN-region gradient ALSO update '
                             'patch_embedding — for ablation only. No effect when EEG backbone is frozen.')
    parser.add_argument('--gcn_region_eeg_proj', type=str, default='linear',
                        choices=('linear', 'conv'),
                        help='Projection kind for --gcn_region_eeg_source cbramod. '
                             '"linear" (default): mean-pool T_seg → Linear(200, 768). 154K params. '
                             '"conv": preserve T_seg, Conv1d(200→768, k=3) + GELU + Conv1d(768→768, k=3) '
                             '→ AdaptiveAvgPool over T_seg. ~2.3M params. Learns temporal patterns per '
                             'channel before pooling, more expressive at higher compute / param cost.')
    # ---- Eval-only prediction dump (default off) ----
    parser.add_argument('--dump_affinity', action='store_true', default=False,
                        help='With --dump_predictions: also dump per-sample GCN affinity '
                             '(node-type degrees, modality coupling, EEG region_alpha gate, '
                             'video-vs-EEG fusion gate) to {name}_affinity.jsonl. Gated -> off.')
    parser.add_argument('--dump_predictions', action='store_true', default=False,
                        help='Eval-only: skip training, load --dump_ckpt into the constructed '
                             'model and write per-sample val predictions (jsonl) via test_only. '
                             'Run with --num_gpus 1. Original training path untouched when off.')
    parser.add_argument('--dump_ckpt', type=str, default='',
                        help='Path to best.pt to load for --dump_predictions.')
    parser.add_argument('--dump_name', type=str, default='model',
                        help='Label used in the dumped {name}_predictions.jsonl filename.')
    parser.add_argument('--dump_dir', type=str, default='./logs_compare',
                        help='Output dir for dumped predictions (sample_predictions/ subdir).')

    parser.add_argument("--eeg_backbone_learning_rate", type=float, default=None,
                        help="Separate LR for the EEG backbone (eeg_model.*, excl. classifier). "
                             "Default None = stays in backbone group at --learning_rate (original). "
                             "Set e.g. 2e-5 for LaBraM, which collapses at the 1e-4 that suits CBraMod.")
    parser.add_argument("--head_learning_rate", type=float, default=1e-4)
    parser.add_argument("--gcn_learning_rate", type=float, default=5e-4)

    parser.add_argument("--head_weight_decay", type=float, default=0.01)
    parser.add_argument("--gcn_weight_decay", type=float, default=0.01)

    args = parser.parse_args()

    # Add master information
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.port

    if __name__ == "__main__":
        multi_run_main(args)
        print("TRAINING IS DONE")
