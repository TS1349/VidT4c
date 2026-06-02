import torch
import torch.nn as nn
import torch.nn.functional as F
from .Medformer_EncDec import Encoder, EncoderLayer
from .SelfAttention_Family import MedformerLayer
from .Embed import ListPatchEmbedding


class Medformer_Model(nn.Module):
    def __init__(
        self,
        args,
        output_dim: int,
        enc_in: int,
        seq_len: int = 256,
        d_model: int = 128,
        n_heads: int = 8,
        e_layers: int = 6,
        d_ff: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
        output_attention: bool = False,
        no_inter_attn: bool = False,
        single_channel: bool = False,
        patch_len_list=(2,4,8,8,16,16,16,16,32,32,32,32,32,32,32,32), # Same with PTB-XL dataset config
        augmentations="jitter0.2,scale0.2,drop0.5",
    ):
        super().__init__()
        if isinstance(patch_len_list, str):
            patch_len_list = [int(x) for x in patch_len_list.split(",") if x]

        self.args = args
        self.output_dim = output_dim
        if output_dim[1] == 1:
            self.output_dim = (output_dim[0],)

        self.num_class = output_dim[0] * output_dim[1]

        self.output_attention = output_attention
        # self.enc_in = enc_in
        self.eeg_ch = enc_in
        self.enc_in = d_model
        self.single_channel = single_channel

        stride_list = list(patch_len_list)
        patch_num_list = [int((seq_len - pl) / st + 2) for pl, st in zip(patch_len_list, stride_list)]

        # Embedding
        self.enc_embedding = ListPatchEmbedding(
            enc_in=self.enc_in,
            d_model=d_model,
            seq_len=seq_len,
            patch_len_list=patch_len_list,
            stride_list=stride_list,
            dropout=dropout,
            augmentations=augmentations,
            single_channel=single_channel,
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    MedformerLayer(
                        len(patch_len_list),
                        d_model,
                        n_heads,
                        dropout,
                        output_attention,
                        no_inter_attn,
                    ),
                    d_model,
                    d_ff,
                    dropout,
                    activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )
        # Decoder
        self.act = F.gelu
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(
            d_model
            * len(patch_num_list)
            * (1 if not self.single_channel else self.enc_in),
            self.num_class,
        )
        self.eeg_reduce = nn.Conv2d(self.eeg_ch, 1, kernel_size=3, stride=1, padding=1, bias=True)

    def classification(self, x_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        if self.single_channel:
            enc_out = torch.reshape(enc_out, (-1, self.enc_in, *enc_out.shape[-2:]))

        # Output
        output = self.act(
            enc_out
        )
        output = self.dropout(output)
        output = output.reshape(
            output.shape[0], -1
        )
        output = self.projection(output)
        return output

    def forward(self, sample):
        eeg = sample["eeg"]
        if self.args.fft_mode == 'Spectrogram':
            eeg = self.eeg_reduce(eeg).squeeze(1)
        output = self.classification(eeg)

        if self.args.dataset in ('emognition', 'mdmer'):
            output_v = output[:, :self.output_dim[0]].unsqueeze(-1)
            output_a = output[:, self.output_dim[0]:].unsqueeze(-1)
            output = torch.concat((output_v, output_a), dim=-1)

        return output  # [B, N]
