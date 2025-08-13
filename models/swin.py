import torch
import torch.nn as nn
import torch.nn.functional as F
from .tsf.layers import Mlp
# from torchvision.models.video.swin_transformer import PatchEmbed3d, SwinTransformer3d
# https://docs.pytorch.org/vision/main/models/generated/torchvision.models.video.swin3d_b.html#torchvision.models.video.swin3d_b
from torchvision.models.video.swin_transformer import PatchEmbed3d, swin3d_b


def patch_embed_3c(
    patch_size,
    embed_dim,
    norm_layer):
    return PatchEmbed3d(
            patch_size = patch_size,
            in_channels = 3,
            embed_dim = embed_dim,
            norm_layer = norm_layer
    )

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

# def get_base_4c(output_dim):
#     return SwinTransformer3d(
#         patch_size=[2, 4, 4],
#         embed_dim=128,
#         depths=[2, 2, 18, 2],
#         num_heads=[4, 8, 16, 32],
#         window_size=[8, 7, 7],
#         stochastic_depth_prob=0.1,
#         patch_embed= patch_embed_4c,
#         num_classes= output_dim,
#     )

# def get_base_3c(output_dim):
#     return SwinTransformer3d(
#         patch_size=[2, 4, 4],
#         embed_dim=128,
#         depths=[2, 2, 18, 2],
#         num_heads=[4, 8, 16, 32],
#         window_size=[8, 7, 7],
#         stochastic_depth_prob=0.1,
#         patch_embed= patch_embed_3c,
#         num_classes= output_dim,
#     )


class BridgedVideoSwin4C(nn.Module):
    def __init__(self, args,
                 output_dim,
                 image_size,
                 eeg_channels,
                 frequency_bins,
                 ):

        super().__init__()

        self.output_dim = output_dim
        self.args = args

        if output_dim[1] == 1:
            self.output_dim = (output_dim[0],)

        if self.args.eeg_signal:
            self.image_size = image_size
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
            
            self.patch_emb = patch_embed_4c

        else:
            self.patch_emb = patch_embed_3c


        if self.args.pretrained:
            weight = "Swin3D_B_Weights.KINETICS400_V1"
        else:
            weight = None

        self.video_model = swin3d_b(weights=weight,
                patch_embed=self.patch_emb,
                num_classes=output_dim[0]*output_dim[1])
    
    def forward(self, x):
        # Choose video vs video+EEG (AbsFFT or STFT) by arg option
        if self.args.eeg_signal:
            if self.args.fft_mode == "AbsFFT":
                eeg = x["eeg"].flatten(start_dim=-2)
                eeg = self.bridge(eeg)
                new_shape = eeg.shape[:2] + (1, self.image_size, self.image_size)
                eeg = eeg.view(new_shape)
            elif self.args.fft_mode == "Spectrogram": 
                eeg = self.eeg2fps(x["eeg"]).unsqueeze(2)
            
            four_channel_video = torch.cat([x["video"], eeg], dim = -3)
            four_channel_video.transpose_(-3, -4)
            output = self.video_model(four_channel_video)
        else:
            output = self.video_model(x["video"].transpose_(-3, -4))

        if self.args.dataset == 'emognition' or 'mdmer':
            output_v = output[:, :self.output_dim[0]].unsqueeze(-1)
            output_a = output[:, self.output_dim[0]:].unsqueeze(-1)
            output = torch.concat((output_v, output_a), dim=-1)

        return output
