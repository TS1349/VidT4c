import torch
import torch.nn as nn
import torch.nn.functional as F
from .Embed import DataEmbedding
from .SelfAttention_Family import FullAttention, AttentionLayer
from .Autoformer_EncDec import (
    Encoder,
    EncoderLayer,
    my_Layernorm,
    series_decomp,
)


class FEDformer_Model(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    Paper link: https://proceedings.mlr.press/v162/zhou22g.html
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
        """
        version: str, for FEDformer, there are two versions to choose, options: [Fourier, Wavelets].
        mode_select: str, for FEDformer, there are two mode selection method, options: [random, low].
        modes: int, modes to be selected.
        """
        super().__init__()
        self.args = args
        self.output_dim = output_dim
        if output_dim[1] == 1:
            self.output_dim = (output_dim[0],)

        self.num_class = output_dim[0] * output_dim[1]

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
                    attention=AttentionLayer(
                        FullAttention(
                            False, factor,
                            attention_dropout=dropout,
                            output_attention=output_attention
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )

        self.act = F.gelu
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(
            d_model * seq_len, self.num_class
        )
        self.eeg_reduce = nn.Conv2d(self.eeg_ch, 1, kernel_size=3, stride=1, padding=1, bias=True)


    def classification(self, x_enc):
        # enc
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)
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
            output_a = output[:, self.output_dim[0]:].unsqueeze(-1)
            output = torch.concat((output_v, output_a), dim=-1)

        return output  # [B, N]