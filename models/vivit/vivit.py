import torch
import torch.nn as nn
import torch.nn.functional as F
from .vit import ViViT
from transformers import VivitConfig, VivitForVideoClassification

class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.,
                 ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class BridgedViViT4C(nn.Module):
    def __init__(self, args,
                 output_dim,
                 image_size,
                 eeg_channels,
                 frequency_bins,
                 ):

        super().__init__()
        self.args = args

        self.image_size = image_size
        self.output_dim = output_dim
        self.num_classes = output_dim[0] * output_dim[1]

        if self.args.eeg_signal:
            if self.args.fft_mode == "AbsFFT":
                out_dim =  self.image_size ** 2
                hidden_dim = out_dim // 2

                self.bridge = Mlp(
                    in_features= eeg_channels * frequency_bins,
                    hidden_features = hidden_dim,
                    out_features = out_dim
                )
            elif self.args.fft_mode == "Spectrogram": 
                self.eeg2fps = nn.Conv2d(eeg_channels, 32, kernel_size=3, stride=1, padding=1, bias=True)

            self.input_ch = 4
        else:
            self.input_ch = 3

        # Using hugging face ViViT model (inclusing pretrained weight)
        vivit_config = VivitConfig(
            image_size=image_size,
            num_frames=32,
            tubelet_size=[2, 16, 16],
            video_size = [32, 224, 224],
            num_channels=self.input_ch,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_labels=self.num_classes,
        )

        if self.args.pretrained:
            self.model = VivitForVideoClassification.from_pretrained(
                "google/vivit-b-16x2-kinetics400",
                config=vivit_config,
                ignore_mismatched_sizes=True,
                force_download=True,
                use_safetensors=True)
        else:
            self.model = VivitForVideoClassification(
            config=vivit_config,)
            

    def forward(self, sample):
        # Choose video vs video+EEG by arg option
        if self.args.eeg_signal:
            if self.args.fft_mode == "AbsFFT":
                eeg = sample["eeg"].flatten(start_dim=-2)
                eeg = self.bridge(eeg)
                new_shape = eeg.shape[:2] + (1, self.image_size, self.image_size)
                eeg = eeg.view(new_shape)
            elif self.args.fft_mode == "Spectrogram": 
                eeg = self.eeg2fps(sample["eeg"]).unsqueeze(2)
            
            four_channel_video = torch.cat([sample["video"], eeg], dim = -3)
            output = self.model(four_channel_video)
        else:
            output = self.model(sample["video"])

        logits = output.logits
        if self.args.dataset in ('emognition', 'mdmer'):
            logits_v = logits[:, :self.output_dim[0]].unsqueeze(-1)
            logits_a = logits[:, self.output_dim[0]:].unsqueeze(-1)
            logits = torch.concat((logits_v, logits_a), dim=-1)
        
        return logits