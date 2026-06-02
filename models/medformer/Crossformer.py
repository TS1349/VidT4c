import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from .Crossformer_EncDec import scale_block, Encoder, Decoder, DecoderLayer
from .Embed import PatchEmbedding
from .SelfAttention_Family import (
    AttentionLayer,
    FullAttention,
    TwoStageAttentionLayer,
)

from math import ceil


class Crossformer_Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=vSVLM2j9eie
    """
    def __init__(
        self,
        args,
        output_dim: tuple,
        enc_in: int,
        seq_len: int = 256,
        d_model: int = 128,
        patch_len: int = 16,
        n_heads: int = 8,
        e_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
        factor: int = 5,
        output_attention: bool = True,
    ):
        super().__init__()
        self.args = args

        self.output_dim = output_dim
        if output_dim[1] == 1:
            self.output_dim = (output_dim[0],)

        self.num_class = output_dim[0] * output_dim[1]

        # self.enc_in = enc_in
        self.eeg_ch = enc_in
        self.enc_in = d_model
        self.seq_len = seq_len
        self.seg_len = 12
        self.win_size = 2
        self.pred_len = 96

        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * seq_len / self.seg_len) * self.seg_len
        self.pad_out_len = ceil(1.0 * self.pred_len / self.seg_len) * self.seg_len
        self.in_seg_num = self.pad_in_len // self.seg_len
        self.out_seg_num = ceil(
            self.in_seg_num / (self.win_size ** (e_layers - 1))
        )
        self.head_nf = d_model * self.out_seg_num

        # Embedding
        self.enc_value_embedding = PatchEmbedding(
            d_model,
            self.seg_len,
            self.seg_len,
            self.pad_in_len - seq_len,
            0,
        )
        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, self.enc_in, self.in_seg_num, d_model)
        )
        self.pre_norm = nn.LayerNorm(d_model)

        # Encoder
        self.encoder = Encoder(
            [
                scale_block(
                    1 if l is 0 else self.win_size,
                    d_model,
                    n_heads,
                    d_ff,
                    1,
                    dropout,
                    output_attention,
                    self.in_seg_num
                    if l is 0
                    else ceil(self.in_seg_num / self.win_size**l),
                    factor,
                )
                for l in range(e_layers)
            ]
        )
        # Decoder
        self.dec_pos_embedding = nn.Parameter(
            torch.randn(
                1, self.enc_in, (self.pad_out_len // self.seg_len), d_model
            )
        )

        self.decoder = Decoder(
            [
                DecoderLayer(
                    TwoStageAttentionLayer(
                        (self.pad_out_len // self.seg_len),
                        factor,
                        d_model,
                        n_heads,
                        output_attention,
                        d_ff,
                        dropout,
                    ),
                    AttentionLayer(
                        FullAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                    ),
                    self.seg_len,
                    d_model,
                    d_ff,
                    dropout=dropout,
                    # activation=configs.activation,
                )
                for l in range(e_layers + 1)
            ],
        )
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(
            self.head_nf * self.enc_in, self.num_class
        )
        self.eeg_reduce = nn.Conv2d(self.eeg_ch, 1, kernel_size=3, stride=1, padding=1, bias=True)

    def classification(self, x_enc):
        # embedding
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))

        x_enc = rearrange(
            x_enc, "(b d) seg_num d_model -> b d seg_num d_model", d=n_vars
        )
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out, attns = self.encoder(x_enc)
        # Output from Non-stationary Transformer
        output = self.flatten(enc_out[-1].permute(0, 1, 3, 2))
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

    def forward(self, sample):
        eeg = sample["eeg"]
        if self.args.fft_mode == 'Spectrogram':
            eeg = self.eeg_reduce(eeg).squeeze(1)
        output = self.classification(eeg)

        if self.args.dataset in ('emognition', 'mdmer'):
            output_v = output[:, :self.output_dim[0]].unsqueeze(-1)
            output_a = output[:,  self.output_dim[0]:].unsqueeze(-1)
            output = torch.cat((output_v, output_a), dim=-1)
        return output
