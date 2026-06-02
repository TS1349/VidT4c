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
        self.classifier = nn.Sequential(
            Rearrange('b c s d -> b (c s d)'),
            nn.Linear(self.num_ch * 10 * 200, 10 * 200),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(10 * 200, 200),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(200, self.num_classes),
        )
        self.apply(_weights_init)

    def forward(self, sample, mask=None):
        x = sample["eeg"]
        patch_emb = self.patch_embedding(x, mask)
        feats = self.encoder(patch_emb)

        output = self.classifier(feats)

        if self.args.dataset in ('emognition', 'mdmer'):
            output_v = output[:, :self.output_dim[0]].unsqueeze(-1)
            output_a = output[:, self.output_dim[0]:].unsqueeze(-1)
            output = torch.cat((output_v, output_a), dim=-1)

        return output


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


    def forward(self, x, mask=None):
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
        # print(patch_emb[5, 5, 5, :])
        # print(spectral_emb[5, 5, 5, :])
        patch_emb = patch_emb + spectral_emb

        positional_embedding = self.positional_encoding(patch_emb.permute(0, 3, 1, 2))
        positional_embedding = positional_embedding.permute(0, 2, 3, 1)

        patch_emb = patch_emb + positional_embedding

        return patch_emb


def _weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)



# if __name__ == '__main__':
#
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model = CBraMod_Model(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12,
#                     nhead=8).to(device)
#     model.load_state_dict(torch.load('pretrained_weights/pretrained_weights.pth',
#                                      map_location=device))
#     a = torch.randn((8, 16, 10, 200)).cuda()
#     b = model(a)
#     print(a.shape, b.shape)
