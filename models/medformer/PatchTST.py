import torch
from torch import nn
from .Transformer_EncDec import Encoder, EncoderLayer
from .SelfAttention_Family import FullAttention, AttentionLayer
from .Embed import PatchEmbedding


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class PatchTST_Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
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

        self.enc_in  = enc_in
        self.seq_len = seq_len

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            d_model, patch_len, stride, padding, dropout
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=output_attention,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout,
                    activation,
                )
                for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )

        # Prediction Head
        self.head_nf = d_model * int((seq_len - patch_len) / stride + 2)
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(
            self.head_nf * enc_in, self.num_class
        )

    def classification(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        # x_enc /= stdev
        x_enc = x_enc / stdev # Fix error -jh

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        )
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, sample):
        x = sample["eeg"]
        output = self.classification(x)

        if self.args.dataset in ('emognition', 'mdmer'):
            output_v = output[:, :self.output_dim[0]].unsqueeze(-1)
            output_a = output[:,  self.output_dim[0]:].unsqueeze(-1)
            output = torch.cat((output_v, output_a), dim=-1)
        return output
