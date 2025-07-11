import os
import random
import torch.multiprocessing as mp
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
# from torchaudio.transforms import Spectrogram

from torchvision.transforms.v2 import Compose, Normalize, Resize, ToDtype
import math
import torch.nn as nn
import torch.optim as optim
import numpy as np

from world_info import init_distributed_mode
from trainer import PTrainer
from models import BridgedTimeSFormer4C, BridgedViViT4C, BridgedVideoSwin4C
from dataloader import EAVDataset, EmognitionDataset, MDMERDataset
from dataloader.transforms import AbsFFT, sftf
from scheduler import CosineScheduler

import argparse

def get_torch_model(name):
    if name == "vivit":
        return BridgedViViT4C
    elif name == "tsf":
        return BridgedTimeSFormer4C
    elif name == "swin":
        return BridgedVideoSwin4C
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
    experiment_name = args.experiment_name
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
    video_preprocessor = Compose(
        [
         Resize(size=(img_size,img_size), antialias=True),
         ToDtype(torch.float32),
        #  ToDtype(torch.float32, scale=True),
         Normalize(
             mean=(0.45, 0.45, 0.45),
             std=(0.225, 0.225, 0.225),
         )]
    )

    # Check the right fourier transformation
    if fft_mode == "AbsFFT":
        fft = AbsFFT(dim=-2)
    # elif fft_mode == "Spectrogram": # Not considering now
    #     fft = sftf(dim=-1)

    training_dataset = torch_dataset(
        csv_file=csv_file,
        time_window = 5.0, #sec
        video_transform=video_preprocessor,
        eeg_transform = fft,
        split = "test"
    )
    validation_dataset = torch_dataset(
        csv_file=csv_file,
        time_window = 5.0, #sec
        video_transform=video_preprocessor,
        eeg_transform = fft,
        split = "test"
    )

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
    )

    training_dataloader = DataLoader(
        dataset=training_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        sampler=training_sampler, drop_last=True, # Add drop_last
    )

    # Val-dataloader shouldn't be sampled by training_sampler
    validation_dataloader = DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        sampler=validation_sampler, drop_last=False,
    )

    model = torch_model(args,
                 output_dim = training_dataset.output_shape,
                 image_size = img_size,
                 eeg_channels = training_dataset.eeg_channel_count,
                 frequency_bins = 64,
    )

    device = torch.device("cuda")
    model = model.to(device)

    loss_function = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
)

    steps_per_epoch = math.ceil(len(training_dataset) / batch_size)
    total_steps = steps_per_epoch * epochs

    lr_scheduler = CosineScheduler(
        optimizer=optimizer,
        t_total=total_steps,
    )

    p_trainer = PTrainer(
        model=model,
        lr_scheduler = lr_scheduler,
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
    print("Process group destroyed; Training finished")


if "__main__" == __name__:

    # TODO
    ## Skip the missing video sequence in EAV dataset -> I'll add it during the weekend

    parser = argparse.ArgumentParser(description="TimeSFormer")

    parser.add_argument("--epochs", type=int, default=4) # Check the code's feasibility
    parser.add_argument("--num_gpus", type=int, default=2) # Check multi-gpu
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0004)
    parser.add_argument("--csv_file", type=str, default= "./datasets/updated_fold_csv_files/MDMER_fold_csv/MDMER_dataset_updated_fold0.csv")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--experiment_name", type=str, default="mdmer_swin_b16_e4_res112_no-eeg")
    parser.add_argument("--dataset", type=str, default="mdmer")
    parser.add_argument("--model", type=str, default="swin")
    parser.add_argument("--port", type=str, default="8008")
    parser.add_argument("--seed", type=int, default="7254")
    parser.add_argument("--img_size", type=int, default="112") # In the Nef resource

    parser.add_argument('--eeg_signal', action='store_true', default=False,
                        help='Choose video or video+EEG by arg option')
    parser.add_argument('--fft_mode', type=str, default='AbsFFT', # Consider this on next step
                        choices=('AbsFFT', 'Spectrogram'), help='Choose FFT transformation method')

    args = parser.parse_args()

    # Add master information
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

    print("TRAINING IS DONE")