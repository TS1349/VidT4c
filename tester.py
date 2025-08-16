import torch
from models import BridgedViViT4C
from collections import namedtuple

ModelArgs = namedtuple("ModelArgs", ["eeg_signal", "fft_model"])

def zero_out_model(model):
    for param in model.parameters():
        param.data.zero_()

def print_weight_sums(model):
    for name, param in model.named_parameters():
        print(f"{name} : {param.data.abs().sum()}")

def print_weight_file(weight):
    for name, param in weight.items():
        print(f"{name} : {param.data.abs().sum()}")


args = ModelArgs(
    eeg_signal=False,
    fft_model="Spectrogram"
)


vivit = BridgedViViT4C(args,
                       output_dim=(2,4),
                       image_size=224,
                       eeg_channels = 0,
                       frequency_bins=0)

zero_out_model(vivit)

print("After zeroing out")
print_weight_sums(vivit)

vivit_weights = torch.load("./pretrained/vivit.pth")

print("Weights file")
print_weight_file(vivit_weights)

vivit.model.load_state_dict(vivit_weights, strict=False)

print("After load")
print_weight_sums(vivit)
