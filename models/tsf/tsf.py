from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from .vit import VisionTransformer
from .layers import Mlp
import os


class BridgedTimeSFormer4C(nn.Module):
    def __init__(self, args,
                 output_dim,
                 image_size,
                 eeg_channels,
                 frequency_bins,
                 ):

        super().__init__()

        self.output_dim = output_dim
        self.image_size = image_size
        self.args = args

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

        # TimeSformer (https://github.com/facebookresearch/TimeSformer/blob/a5ef29a7b7264baff199a30b3306ac27de901133/timesformer/models/vit.py)
        self.video_model = VisionTransformer(
            img_size=image_size,
            output_dim= output_dim[0]*output_dim[1],
            patch_size=16,
            in_chans=self.input_ch,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            num_frames=32,
            attention_type='divided_space_time',
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
            output = self.video_model(four_channel_video)
        else:
            output = self.video_model(x["video"].transpose_(-3, -4))

        new_out_shape = output.shape[:-1] + self.output_dim
        output = output.view(new_out_shape)
        return output
