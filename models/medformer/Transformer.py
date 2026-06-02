import torch
import torch.nn as nn
import torch.nn.functional as F
from .Transformer_EncDec import (
    Encoder,
    EncoderLayer,
)
from .SelfAttention_Family import FullAttention, AttentionLayer
from .Embed import DataEmbedding


class EEG_Transformer(nn.Module):
    """
    Vanilla Transformer - classification only
    Input:  x_enc (B, seq_len, enc_in)
    Output: (B, num_class)
    """
    def __init__(
        self,
        args,
        output_dim: tuple,
        enc_in: int,
        seq_len: int = 256,
        d_model: int = 128,
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

        self.enc_in  = enc_in
        self.seq_len = seq_len

        # Embedding
        self.enc_embedding = DataEmbedding(
            c_in=enc_in,
            d_model=d_model,
            embed_type="timeF",
            freq="h",
            dropout=dropout,
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            factor,
                            dropout,
                            output_attention,
                        ),
                        d_model,
                        n_heads,
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

        # Decoder (head)
        self.act = F.gelu
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(d_model * seq_len, self.num_class)

    def classification(self, x_enc):
        # x_enc: (B, L, C)
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        out = self.dropout(self.act(enc_out))
        out = out.reshape(out.shape[0], -1)
        out = self.projection(out)
        return out

    def forward(self, sample):
        x = sample["eeg"]
        output = self.classification(x)

        if self.args.dataset in ('emognition', 'mdmer'):
            output_v = output[:, :self.output_dim[0]].unsqueeze(-1)
            output_a = output[:,  self.output_dim[0]:].unsqueeze(-1)
            output = torch.cat((output_v, output_a), dim=-1)
        return output
