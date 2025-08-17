import torch
from models import BridgedViViT4C, BridgedTimeSFormer4C, BridgedVideoSwin4C
from collections import namedtuple

ModelArgs = namedtuple("ModelArgs", ["eeg_signal", "fft_mode", "pretrained"])

def zero_out_model(model):
    for param in model.parameters():
        param.data.zero_()

def print_weight_sums(model):
    for name, param in model.named_parameters():
        print(f"{name} : {param.data.abs().sum()}")

def print_weight_file(weight):
    for name, param in weight.items():
        if isinstance(param,int):
            print(f"{name} : {param}")
        else:
            print(f"{name} : {param.data.abs().sum()}")


args = ModelArgs(
    eeg_signal=True,
    fft_mode="Spectrogram",
    pretrained=True,
)

model_name = "swin"


if "vivit" == model_name:
    model = BridgedViViT4C(args,
                           output_dim=(2,4),
                           image_size=224,
                           eeg_channels = 0,
                           frequency_bins=0)
elif "tsf" == model_name:
    model = BridgedTimeSFormer4C(args,
                                 output_dim=(2,4),
                                 image_size=224,
                                 eeg_channels = 0,
                                 frequency_bins=0)
elif "swin" == model_name:
    model = BridgedVideoSwin4C(args,
                                 output_dim=(2,4),
                                 image_size=224,
                                 eeg_channels = 0,
                                 frequency_bins=0)

else:
    raise Exception("Wrong name")

print_weight_sums(model)
#zero_out_model(model)
#
#print("After zeroing out")
#print_weight_sums(model)

# weights = torch.load(f"./pretrained/{model_name}.pth", weights_only=True)

#print("Weights file")
# print_weight_file(weights)

#model.model.load_state_dict(weights, strict=False)

#print("After load")
#print_weight_sums(model)
