import torch
import torch.nn as nn
import torch.nn.functional as F
from .vit import ViViT
import os

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
        if output_dim[1] == 1:
            self.output_dim = (output_dim[0],)

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
                self.eeg2fps = nn.Conv2d(eeg_channels * 2, 32, kernel_size=3, stride=1, padding=1, bias=True)

            self.input_ch = 4
        else:
            self.input_ch = 3

        # ViViT (https://github.com/rishikksh20/ViViT-pytorch)
        self.model = ViViT(
            image_size=image_size,
            patch_size=16,
            num_classes=output_dim[0] * output_dim[1],
            num_frames=32,
            dim=768,
            depth=12,
            heads=12,
            pool='cls',
            in_channels=self.input_ch,
            dim_head=64,
            dropout=0.,
            emb_dropout=0.,
            scale_dim=4,
            tubelet_size=2,
        )

    
    def forward(self, x):
        # Choose video vs video+EEG by arg option
        if self.args.eeg_signal:
            if self.args.fft_mode == "AbsFFT":
                eeg = x["eeg"].flatten(start_dim=-2)
                eeg = self.bridge(eeg)
                new_shape = eeg.shape[:2] + (1, self.image_size, self.image_size)
                eeg = eeg.view(new_shape)
            elif self.args.fft_mode == "Spectrogram": 
                eeg = x["eeg"]
                eeg = F.interpolate(eeg, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
                eeg = self.eeg2fps(eeg).unsqueeze(2)
            
            four_channel_video = torch.cat([x["video"], eeg], dim = -3)
            four_channel_video.transpose_(-3, -4)
            output = self.model(four_channel_video)
        else:
            output = self.model(x["video"].transpose_(-3, -4))

        new_out_shape = output.shape[:-1] + self.output_dim
        output = output.view(new_out_shape)
        return output
