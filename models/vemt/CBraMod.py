import torch
import torch.nn as nn
import torch.nn.functional as F

from .criss_cross_transformer import TransformerEncoderLayer, TransformerEncoder
from einops.layers.torch import Rearrange

class CBraMod_Model(nn.Module):
    def __init__(self, args, in_dim=200, output_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12,
                    nhead=8, in_chans = int):
        super(CBraMod_Model, self).__init__()

        self.num_ch = in_chans
        self.args = args
        self.output_dim = output_dim
        if output_dim[1] == 1:
            self.output_dim = (output_dim[0],)

        self.num_classes = output_dim[0] * output_dim[1]

        self.patch_embedding = PatchEmbedding(in_dim, self.num_classes, d_model, seq_len)
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, norm_first=True,
            activation=F.gelu
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=n_layer, enable_nested_tensor=False)

        # Same with the original repo -> proj_out: identity -> classifier to downstream task
        # if self.args.set_eeg_only or self.args.fusion == 'router':
        self.classifier = nn.Sequential(
            Rearrange('b c s d -> b (c s d)'),
            nn.Linear(self.num_ch * 10 * 200, 10 * 200),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(10 * 200, 768),
        )
    # elif self.args.fusion == 'router':
        self.classifier2 = nn.Sequential(
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(768, self.num_classes),
        )
        # else:
        #     self.classifier = nn.Sequential(
        #         Rearrange('b c s d -> b (c s d)'),
        #         nn.Linear(self.num_ch * 10 * 200, 10 * 200),
        #         nn.ELU(),
        #         nn.Dropout(0.1),
        #         nn.Linear(10 * 200, 768),
        #     )
        self.apply(_weights_init)

    def forward(self, sample, mask=None, return_encoder_feats=False, return_per_channel_pre=False):
        x = sample["eeg"]
        if return_per_channel_pre:
            patch_emb, spectral_emb, per_ch_pre = self.patch_embedding(x, mask, return_pre_pos=True)
        else:
            patch_emb, spectral_emb = self.patch_embedding(x, mask)
            per_ch_pre = None
        feats = self.encoder(patch_emb)  # [B, ch, T_seg, D]

        feats_0 = self.classifier(feats)
        output = self.classifier2(feats_0)

        # if self.args.set_eeg_only or self.args.fusion == 'router':
        if self.args.dataset in ('emognition', 'mdmer'):
            output_v = output[:, :self.output_dim[0]].unsqueeze(-1)
            output_a = output[:, self.output_dim[0]:].unsqueeze(-1)
            output = torch.cat((output_v, output_a), dim=-1)

        # Variadic return: keep base (output, feats_0); append extras in fixed order.
        extras = []
        if return_encoder_feats:
            extras.append(feats)   # [B, ch, T_seg, D] — post-encoder, channel-mixed
        if return_per_channel_pre:
            extras.append(per_ch_pre)  # [B, ch, T_seg, D] — pre-positional, channel-independent
        if extras:
            return (output, feats_0, *extras)
        return output, feats_0

class PatchEmbedding(nn.Module):
    def __init__(self, in_dim, num_class, d_model, seq_len):
        super().__init__()
        self.d_model = d_model
        self.positional_encoding = nn.Sequential(
            nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(19, 7), stride=(1, 1), padding=(9, 3),
                      groups=d_model),
        )
        self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=False)
        # self.mask_encoding = nn.Parameter(torch.randn(in_dim), requires_grad=True)

        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)),
            nn.GroupNorm(5, 25),
            nn.GELU(),

            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),

            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
        )
        self.spectral_proj = nn.Sequential(
            nn.Linear(101, d_model),
            nn.Dropout(0.1),
            # nn.LayerNorm(d_model, eps=1e-5),
        )

        # self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        # self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        # self.proj_in = nn.Sequential(
        #     nn.Linear(in_dim, d_model, bias=False),
        # )


    def forward(self, x, mask=None, return_pre_pos=False):
        """Patch embedding forward.

        Args:
            return_pre_pos: if True, also return the per-channel feature snapshot
                taken AFTER (proj_in conv + spectral FFT projection) but BEFORE
                the positional encoding's (19, 7) conv (which mixes neighboring
                channels). That snapshot is the last point at which channels
                are independent of each other — useful for downstream modules
                (e.g. GCN region nodes) that want pure per-channel features.
                Shape: [B, ch_num, patch_num, d_model].
        """
        bz, ch_num, patch_num, patch_size = x.shape
        if mask == None:
            mask_x = x
        else:
            mask_x = x.clone()
            mask_x[mask == 1] = self.mask_encoding

        mask_x = mask_x.contiguous().view(bz, 1, ch_num * patch_num, patch_size)
        patch_emb = self.proj_in(mask_x)
        patch_emb = patch_emb.permute(0, 2, 1, 3).contiguous().view(bz, ch_num, patch_num, self.d_model)

        mask_x = mask_x.contiguous().view(bz*ch_num*patch_num, patch_size)
        spectral = torch.fft.rfft(mask_x, dim=-1, norm='forward')
        spectral = torch.abs(spectral).contiguous().view(bz, ch_num, patch_num, 101)
        spectral_emb = self.spectral_proj(spectral)

        patch_emb = patch_emb + spectral_emb
        pre_pos = patch_emb if return_pre_pos else None  # snapshot the per-channel feature

        positional_embedding = self.positional_encoding(patch_emb.permute(0, 3, 1, 2))
        positional_embedding = positional_embedding.permute(0, 2, 3, 1)

        patch_emb = patch_emb + positional_embedding

        if return_pre_pos:
            return patch_emb, spectral_emb, pre_pos
        return patch_emb, spectral_emb


def _weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

