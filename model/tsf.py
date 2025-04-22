from functools import partial
import torch
import torch.nn as nn
from .vit import VisionTransformer
from .layers import Mlp


class BridgedTimeSFormer4C(nn.Module):
    def __init__(self,
                 output_dim,
                 image_size,
                 eeg_channels,
                 frequency_bins,
                 ):

        super().__init__()

        self.output_dim = output_dim


        self.video_model = VisionTransformer(
            img_size=image_size,
            output_dim= output_dim[0]*output_dim[1],
            patch_size=16,
            in_chans=4,
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
        if output_dim[1] == 1:
            self.output_dim = (output_dim[1],)


        self.image_size = image_size
        out_dim =  self.image_size ** 2
        hidden_dim = out_dim // 2

        self.bridge = Mlp(
            in_features= eeg_channels * frequency_bins,
            hidden_features = hidden_dim,
            out_features = out_dim
        )
    
    def forward(self, x):
        eeg = x["eeg"].flatten(start_dim=-2)
        eeg = self.bridge(eeg)
        new_shape = eeg.shape[:2] + (1, self.image_size, self.image_size)
        eeg = eeg.view(new_shape)
        four_channel_video = torch.cat([x["video"], eeg], dim = -3)
        four_channel_video.transpose_(-3, -4)
        output = self.video_model(four_channel_video)

        new_out_shape = output.shape[:-1] + self.output_dim
        output = output.view(new_out_shape)
        return output
