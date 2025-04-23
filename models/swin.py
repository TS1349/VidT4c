import torch
import torch.nn as nn
from .tsf.layers import Mlp
from torchvision.models.video.swin_transformer import PatchEmbed3d, SwinTransformer3d

def patch_embed_4c(
    patch_size,
    embed_dim,
    norm_layer):
    return PatchEmbed3d(
            patch_size = patch_size,
            in_channels = 4,
            embed_dim = embed_dim,
            norm_layer = norm_layer
    )

def get_base_4c(output_dim):
    return SwinTransformer3d(
        patch_size=[2, 4, 4],
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=[8, 7, 7],
        stochastic_depth_prob=0.1,
        patch_embed= patch_embed_4c,
        num_classes= output_dim,
    )

class BridgedVideoSwin4C(nn.Module):
    def __init__(self,
                 output_dim,
                 image_size,
                 eeg_channels,
                 frequency_bins,
                 ):

        super().__init__()

        self.output_dim = output_dim


        self.video_model = get_base_4c(
            output_dim= output_dim[0]*output_dim[1],
        )
        if output_dim[1] == 1:
            self.output_dim = (output_dim[0],)


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
