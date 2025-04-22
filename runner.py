import idr_torch
import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torchvision.transforms.v2 import Compose, Normalize, Resize, ToDtype
import math
import torch.nn as nn
import torch.optim as optim

from trainer import PTrainer
from models import BridgedTimeSFormer4C, BridgedViViT4C
from dataloader import EAVDataset, EmognitionDataset, MDMERDataset
from dataloader.transforms import AbsFFT
from scheduler import CosineScheduler

import argparse

def get_torch_model(name):
    if name == "vivit":
        return BridgedViViT4C
    elif name == "tsf":
        return BridgedTimeSFormer4C
    else:
        raise Exception("Wrong model name")

def get_torch_dataset(name):
    if name == "mdmer":
        return MDMERDataset
    elif name == "emognition":
        return EmognitionDataset
    elif name == "eav":
        return EAVDataset


def run(
    torch_model,
    torch_dataset,
    epochs,
    batch_size,
    learning_rate,
    weight_decay,
    csv_file,
    experiment_name,
    checkpoint_dir,
):

    init_process_group(backend='nccl',
                        init_method='env://',
                        world_size=idr_torch.size,
                        rank=idr_torch.rank)



    video_preprocessor = Compose(
        [
         Resize(size=(224,224)),
         ToDtype(torch.float32, scale=True),
         Normalize(
             mean=(0.45, 0.45, 0.45),
             std=(0.225, 0.225, 0.225),
         )]
    )

    training_dataset = torch_dataset(
        csv_file=csv_file,
        time_window = 5.0, #sec
        video_transform=video_preprocessor,
        eeg_transform = AbsFFT(dim=-2),
        split = "train"
    )
    validation_dataset = torch_model(
        csv_file=csv_file,
        time_window = 5.0, #sec
        video_transform=video_preprocessor,
        eeg_transform = AbsFFT(dim=-2),
        split = "test"
    )

    training_sampler = DistributedSampler(
        dataset=training_dataset,
        num_replicas=idr_torch.size,
        rank=idr_torch.rank,
        shuffle=True,
    )

    batch_size_per_gpu = batch_size // idr_torch.size
    real_batch_size = batch_size_per_gpu * idr_torch.size
    steps_per_epoch = math.ceil(len(training_dataset) / real_batch_size)
    total_steps = steps_per_epoch * epochs

    if(idr_torch.rank == 0):
        print(f"Batch Size per GPU: {batch_size_per_gpu}")
        print(f"Total trianing steps: {total_steps}")

    training_dataloader = DataLoader(
        dataset=training_dataset,
        batch_size=batch_size_per_gpu,
        shuffle=False,
        pin_memory=True,
        sampler=training_sampler,
    )

    validation_dataloader = DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size_per_gpu,
        pin_memory=True,
        shuffle=False,
    )

    model = torch_model(
                 output_dim = training_dataset.output_shape,
                 image_size = 224,
                 eeg_channels = training_dataset.eeg_channel_count,
                 frequency_bins = 64,
    )

    torch.cuda.set_device(idr_torch.local_rank)
    gpu = torch.device("cuda")
    model = model.to(gpu)


    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
)
    lr_scheduler = CosineScheduler(
            optimizer = optimizer,
            t_total = total_steps,
            )


    p_trainer = PTrainer(
        model=model,
        lr_scheduler = lr_scheduler,
        training_dataloader=training_dataloader,
        validation_dataloader=validation_dataloader,
        optimizer=optimizer,
        loss_function=loss_function,
        checkpoint_dir=checkpoint_dir,
        gpu=gpu,
        gpu_id=idr_torch.local_rank,
        experiment_name=experiment_name,
    )

    p_trainer.train(epochs, save_every = 10)

    destroy_process_group()
    print(f"{dir_torch.rank} process group destroyed")


if "__main__" == __name__:
    parser = argparse.ArgumentParser(description="TimeSFormer")

    parser.add_argument("--epochs", type=int, required = True)
    parser.add_argument("--batch_size", type=int, required = True)
    parser.add_argument("--learning_rate", type=float, required = True)
    parser.add_argument("--weight_decay", type=float, required = True)
    parser.add_argument("--csv_file", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)

    args = parser.parse_args()

    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    csv_file = args.csv_file
    experiment_name = args.experiment_name
    checkpoint_dir =args.checkpoint_dir + r"/" + experiment_name
    torch_model = get_torch_model(args.model)
    torch_dataset = get_torch_dataset(args.dataset)


    run(epochs,
        torch_model,
        torch_dataset,
        batch_size,
        learning_rate,
        weight_decay,
        csv_file,
        experiment_name,
        checkpoint_dir,)

    print("TRAINING IS DONE")
