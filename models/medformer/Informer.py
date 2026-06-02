import torch
import torch.nn as nn
import torch.nn.functional as F
from .Transformer_EncDec import (
    Encoder,
    EncoderLayer,
)
from .SelfAttention_Family import FullAttention, AttentionLayer
from .Embed import DataEmbedding


class Informer_Model(nn.Module):
    """
    Informer with Propspare attention in O(LlogL) complexity
    Paper link: https://ojs.aaai.org/index.php/AAAI/article/view/17325/17132
    """

    # def __init__(self, configs):
    #     super(Model, self).__init__()
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
        stride: int = 8,
        padding: int = 8,
        output_attention: bool = True,
    ):
        super().__init__()
        self.args = args

        self.output_dim = output_dim
        if output_dim[1] == 1:
            self.output_dim = (output_dim[0],)

        self.num_class = output_dim[0] * output_dim[1]

        self.output_attention = output_attention
        # self.enc_in = enc_in
        self.eeg_ch = enc_in
        self.enc_in = d_model


        # Embedding
        self.enc_embedding = DataEmbedding(
            c_in=self.enc_in,
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

        self.act = F.gelu
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(d_model * seq_len, self.num_class)
        self.eeg_reduce = nn.Conv2d(self.eeg_ch, 1, kernel_size=3, stride=1, padding=1, bias=True)


    def classification(self, x_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(
            enc_out
        )  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output.reshape(
            output.shape[0], -1
        )  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
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