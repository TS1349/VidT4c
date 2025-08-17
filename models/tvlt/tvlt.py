import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from timm.models.vision_transformer import PatchEmbed
from timm.models.layers import DropPath
from .heads import Pooler
# from timm.models.registry import register_model



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(
                ~mask[:, None, None, :].bool(), -1e12)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class AudioPatchEmbed(nn.Module):
    """ Audio to Patch Embedding"""

    def __init__(
        self,
        img_size=173,
        patch_size=[16, 16],
        in_chans=1,
        embed_dim=768,
    ):
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class TVLTTransformer(nn.Module):

    def __init__(
        self, args, output_dim, eeg_channels, img_size=224, frames=32, in_chans=3,
        patch_size=16, audio_patch_size=[16, 16], embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm), eps=1e-6,
    ):

        super().__init__()
        self.args = args
        
        self.img_size = img_size
        self.num_frames = frames
        self.max_frames = 64
        self.max_audio_patches = 1020
        self.hidden_size = 768
        self.frequency_size = 128
        self.num_classes = output_dim[0] * output_dim[1]
        self.output_dim = output_dim

        self.eeg_reduce = nn.Conv2d(eeg_channels * 2, 1, kernel_size=3, stride=1, padding=1, bias=True)

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        use_audio = True
        self.use_audio = use_audio
        self.patch_embed_v = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.num_patches_v = self.patch_embed_v.num_patches
        self.type_embed_v = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.patch_size = patch_size
        self.temporal_embed = nn.Parameter(torch.zeros(
            1, self.max_frames, self.hidden_size))
        self.pos_embed_v = nn.Parameter(
            torch.zeros(1, self.num_patches_v, embed_dim))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if use_audio:
            self.patch_embed_a = AudioPatchEmbed(
                img_size=img_size,
                patch_size=audio_patch_size,
                in_chans=1,
                embed_dim=embed_dim,
            )
            self.audio_patch_size = audio_patch_size
            self.type_embed_a = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed_a = nn.Parameter(torch.zeros(
                1, self.max_audio_patches, embed_dim))
            self.freq_patch_size = self.frequency_size //audio_patch_size[1]
            self.freq_embed = nn.Parameter(torch.zeros(
                1, self.freq_patch_size, self.hidden_size))

        self.norm = norm_layer(embed_dim)

        self.num_frames = self.num_frames
        self.max_audio_patches = self.max_audio_patches
        self.frame_masking = False

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        hs = self.hidden_size

        # ===================== Downstream ===================== #

        self.classifier = nn.Sequential(
            Pooler(self.hidden_size),
            nn.Linear(hs, hs * 2),
            nn.LayerNorm(hs * 2),
            nn.GELU(),
            nn.Linear(hs * 2, self.num_classes),
        )
        self.classifier.apply(self.init_weights_classify)

    def init_weights_classify(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed_v", "pos_embed_a", "cls_token", "temporal_embed"}


    def random_masking_audio(self, x, att_mask=None, mask_ratio=0.15):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """

        N, L, D = x.shape  # batch, length, dim
        F, T = 8, L//8  # frequency, time
        len_keep = int(L * (1 - mask_ratio))
        # noise in [0, 1]
        noise = torch.rand(N, T, device=x.device).unsqueeze(-1).repeat(1, 1, F).view(N, L)

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        if att_mask is not None:
            mask *= att_mask

        att_mask = torch.gather(att_mask, dim=1, index=ids_keep)
        return x_masked, mask, ids_restore, att_mask

    def random_masking(self, x, att_mask=None, mask_ratio=0.75):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        if att_mask is not None:
            mask *= att_mask

        att_mask = torch.gather(att_mask, dim=1, index=ids_keep)
        return x_masked, mask, ids_restore, att_mask

    def cat_mask(self, mask_token, x, ids_restore):
        mask_tokens = mask_token.repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        return x_

    def get_patch_mask(self, x):
        """
        masks out blank regions of the audios/images.
        """
        if len(x.shape) == 5:
            x = x.mean(2)
            x = F.avg_pool2d(x, self.patch_size,
                             self.patch_size).flatten(2).flatten(1)
            x_mask = x != -1
            return x_mask
        else:
            x = x.mean(1)
            x = F.avg_pool2d(x, self.audio_patch_size,
                             self.audio_patch_size).flatten(1)
            x_mask = x != -1
            return x_mask

    def forward(self, sample):
        video = sample["video"]
        eeg = sample["eeg"]
        # eeg = F.interpolate(eeg, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        eeg = F.interpolate(eeg, size=(256, 128), mode='bilinear', align_corners=False)
        
        eeg = self.eeg_reduce(eeg)

        if eeg is not None:
            x_a = self.patch_embed_a(eeg)
            x_a += self.freq_embed.repeat(1, x_a.size(1)//self.freq_patch_size, 1)
            x_a += torch.repeat_interleave(self.pos_embed_a[:, :x_a.size(
                1)//self.freq_patch_size], self.freq_patch_size, dim=1)
            x_a += self.type_embed_a
            full_x_mask_a = self.get_patch_mask(eeg)

        if video is not None:
            b, t, c, h, w = video.shape
            x_v = self.patch_embed_v(video.reshape(b*t, c, h, w))
            x_v = x_v.reshape(b, t * x_v.size(1), x_v.size(-1))
            frame_patch_len = x_v.size(1)//t
            x_v += self.pos_embed_v.repeat(1, t, 1)
            x_v += torch.repeat_interleave(
                self.temporal_embed[:, :self.num_frames], frame_patch_len, dim=1)
            x_v += self.type_embed_v
            full_x_mask_v = self.get_patch_mask(video)

        if eeg is not None and video is not None:
            enc_mask = torch.cat(
                [full_x_mask_a[:, :1], full_x_mask_a, full_x_mask_v], 1)
            x = torch.cat([self.cls_token.expand(
                x_v.size(0), -1, -1), x_a, x_v], 1)
        elif eeg is not None:
            enc_mask = full_x_mask_a
            x = x_a

        for blk in self.blocks:
            x = blk(x, enc_mask)
        x = self.norm(x)

        # Add classifier head to origin code
        output = self.classifier(x)

        if self.args.dataset in ('emognition', 'mdmer'):
            output_v = output[:, :self.output_dim[0]].unsqueeze(-1)
            output_a = output[:, self.output_dim[0]:].unsqueeze(-1)
            output = torch.concat((output_v, output_a), dim=-1)

        return output