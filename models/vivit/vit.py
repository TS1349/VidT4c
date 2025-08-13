import torch
from torch import nn
from einops import rearrange, repeat
from .module import Attention, PreNorm, FeedForward
from .weight import trunc_normal_, constant_init_, kaiming_init_

# === PatchEmbed fix ===
class PatchEmbed(nn.Module):
    """ Tubelet Embedding Module """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 num_frames=16, tubelet_size=2):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.tubelet_size = int(tubelet_size)
        
        # for reference only, not strictly used in forward
        num_patches = (img_size[1] // patch_size[1]) \
                      * (img_size[0] // patch_size[0]) \
                      * (num_frames // self.tubelet_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        # 3D conv kernel = (tubelet, patchH, patchW)
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(self.tubelet_size, patch_size[0], patch_size[1]),
            stride=(self.tubelet_size, patch_size[0], patch_size[1])
        )

    def forward(self, x):
        """
        Expects x of shape [B, in_chans=4, T, H, W].
        After conv3d => out shape is [B, embed_dim, T', H', W'].
        We'll reshape to [B, T', H'*W', embed_dim], i.e. 4D => return that.
        """
        B, C, T, H, W = x.shape
        # safety check
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}x{W}) doesn't match model ({self.img_size[0]}x{self.img_size[1]})."

        # [B, embed_dim, T', H', W']
        x = self.proj(x)

        # Let's reshape to 4D
        # Suppose x is [B, embed_dim=192, T'=(T/tubelet), H'=(H/patch), W'=(W/patch)]
        B, EMBED, T_, H_, W_ = x.shape
        # Now we want [B, T_, (H_*W_), EMBED], i.e. 4D
        x = x.reshape(B, EMBED, T_, H_*W_)  # => [B, EMBED, T_, H_*W_]
        x = x.permute(0, 2, 3, 1)          # => [B, T_, H_*W_, EMBED]

        return x  # => shape [b, t, n, dim]


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class ViViT(nn.Module):
    def __init__(self,
                 image_size,
                 patch_size,
                 num_classes,
                 num_frames,
                 dim=192,
                 depth=4,
                 heads=3,
                 pool='mean', 
                 in_channels=3,
                 dim_head=64,
                 dropout=0.,
                 emb_dropout=0.,
                 scale_dim=4,
                 tubelet_size=2,
                 ):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = PatchEmbed(
            img_size=image_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=dim,
            num_frames=num_frames,
            tubelet_size=tubelet_size
        )

        num_patches_per_frame = (image_size // patch_size) ** 2
        num_frames_after_tubelet = num_frames // tubelet_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames_after_tubelet, num_patches_per_frame + 1, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim * scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.mlp_head = nn.Linear(dim, num_classes)

        # Deeper classifier head (same with TVLT model)
        # self.mlp_head = nn.Sequential(
        #     nn.Linear(dim, dim * 2),
        #     nn.LayerNorm(dim * 2),
        #     nn.GELU(),
        #     nn.Linear(dim * 2, self.num_classes),
        # )

        trunc_normal_(self.pos_embedding, std=.02)
        trunc_normal_(self.space_token, std=.02)
        trunc_normal_(self.temporal_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x):
        x = self.to_patch_embedding(x)  # [batch, time, num_patches, dim]
        b, t, n, _ = x.shape

        cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', b=b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        x += self.pos_embedding[:, :, :(n + 1)]
        x = self.dropout(x)

        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.space_transformer(x)
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)

        cls_temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)

        x = self.temporal_transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        return self.mlp_head(x)

