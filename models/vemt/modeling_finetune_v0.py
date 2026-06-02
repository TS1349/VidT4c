# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from timm.layers import drop_path, to_2tuple, trunc_normal_
import torch.utils.checkpoint as checkpoint

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 400,
        'input_size': (3, 224, 224),
        'pool_size': None,
        'crop_pct': .9,
        'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5),
        'std': (0.5, 0.5, 0.5),
        **kwargs
    }


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


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
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (num_frames // self.tubelet_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim, 
                            kernel_size = (self.tubelet_size,  patch_size[0],patch_size[1]), 
                            stride=(self.tubelet_size,  patch_size[0],  patch_size[1]))

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return  torch.tensor(sinusoid_table,dtype=torch.float, requires_grad=False).unsqueeze(0) 


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """ 

    def __init__(self, args,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 output_dim=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 fc_drop_rate=0., 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False, 
                 init_scale=0.,
                 all_frames=16,
                 tubelet_size=2,
                 use_checkpoint=False,
                 use_mean_pooling=True):
        super().__init__()

        self.args = args
        self.num_classes = output_dim[0] * output_dim[1]
        self.num_value = output_dim[0]
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.tubelet_size = tubelet_size
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=all_frames, tubelet_size=self.tubelet_size)
        num_patches = self.patch_embed.num_patches
        self.use_checkpoint = use_checkpoint

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            # sine-cosine positional embeddings is on the way
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.fc_dropout = nn.Dropout(p=fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        if self.args.set_video_only or self.args.fusion == 'router':
            self.head = nn.Linear(embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

            if use_learnable_pos_emb:
                trunc_normal_(self.pos_embed, std=.02)

            # trunc_normal_(self.head.weight, std=.02)
            # self.apply(self._init_weights)

            # self.head.weight.data.mul_(init_scale)
            # self.head.bias.data.mul_(init_scale)

        self.align_module = LocalAlign(
            d_model=768,
            num_frames=16,
            window_size=1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x, eeg_feat=None, return_tokens=False):
        x = self.patch_embed(x)
        B, _, _ = x.size()

        if self.pos_embed is not None:
            x = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        x = self.pos_drop(x)

        if self.use_checkpoint:
            for blk in self.blocks:
                x = checkpoint.checkpoint(blk, x)
        else:
            for i, blk in enumerate(self.blocks):
                x = blk(x)

        x = self.norm(x)  # nn.Identity() when use_mean_pooling=True
        if return_tokens:
            global_f = self.fc_norm(x.mean(1)) if self.fc_norm is not None else x[:, 0]
            return global_f, x  # [B, 768], [B, N, 768]
        if self.fc_norm is not None:
            return self.fc_norm(x.mean(1))
        else:
            return x[:, 0]

    def forward_features_until(self, x, end_block_idx):
        """Forward up to (excluding) blocks[end_block_idx].

        Returns the input tensor to blocks[end_block_idx], i.e. the cacheable
        intermediate when blocks[end_block_idx:] are the trainable tail.
        """
        x = self.patch_embed(x)
        B, _, _ = x.size()
        if self.pos_embed is not None:
            x = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        x = self.pos_drop(x)
        for blk in self.blocks[:end_block_idx]:
            x = blk(x)
        return x  # [B, num_patches, embed_dim]

    def forward_features_from(self, h, start_block_idx):
        """Resume forward from blocks[start_block_idx] through norm + (fc_norm or CLS).

        Mirrors the tail of forward_features. Returns the global feature [B, D].
        Used to skip frozen blocks when their output is cached.
        """
        for blk in self.blocks[start_block_idx:]:
            h = blk(h)
        h = self.norm(h)
        if self.fc_norm is not None:
            return self.fc_norm(h.mean(1))
        return h[:, 0]

    def forward(self, x, eeg_feat=None, return_feat=False, return_tokens=False):
        result = self.forward_features(x, eeg_feat, return_tokens=return_tokens)

        if return_tokens:
            global_f, tokens = result

            if self.args.set_video_only or self.args.fusion == 'router':
                out = self.head(self.fc_dropout(global_f))
                if self.args.dataset in ('emognition', 'mdmer'):
                    x_v = out[:, :self.num_value].unsqueeze(-1)
                    x_a = out[:, self.num_value:].unsqueeze(-1)
                    out = torch.cat((x_v, x_a), dim=-1)

                if return_feat:
                    return out, global_f, tokens
                return out, tokens

            if return_feat:
                return global_f, global_f, tokens
            return global_f, tokens

        video_f = result  # [B, 768]

        if self.args.set_video_only or self.args.fusion == 'router':
            out = self.head(self.fc_dropout(video_f))
            if self.args.num_clips == 1 and self.args.dataset in ('emognition', 'mdmer'):
                x_v = out[:, :self.num_value].unsqueeze(-1)
                x_a = out[:, self.num_value:].unsqueeze(-1)
                out = torch.cat((x_v, x_a), dim=-1)

            if return_feat:
                return out, video_f
            return out

        if return_feat:
            return video_f, video_f
        return video_f


class LocalAlign(nn.Module):
    def __init__(self, d_model, num_frames=16, window_size=3):
        super().__init__()
        self.num_frames = num_frames   # T'
        self.window_size = window_size

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)

        self.scale = d_model ** -0.5
        # self.scale = 1

    def forward(self, eeg_feat, video_feat_orig):
        """
        eeg_feat:   (B, Te, D)
        video_feat: (B, Nv, D)  # Nv = T' * S
        """

        B, Nv, D = video_feat_orig.shape
        Te = eeg_feat.shape[1]

        T = 32
        S = Nv // T

        # video → (T, S)
        video_feat = video_feat_orig.view(B, T, S, D)

        # spatial pooling → time representation
        video_time = video_feat.mean(dim=2)  # (B, T, D)

        # attention
        Q = self.q_proj(video_time)   # (B, T, D)
        K = self.k_proj(eeg_feat)     # (B, Te, D)
        V = eeg_feat                 # (B, Te, D)

        sim = torch.einsum("btd,bsd->bts", Q, K) * self.scale  # (B, T, Te)

        # time local mask
        if self.window_size is not None:
            idx_v = torch.arange(T, device=video_feat.device)
            idx_e = torch.arange(Te, device=video_feat.device)

            dist = idx_v[:, None] - idx_e[None, :]
            mask = (dist.abs() <= self.window_size)  # (T, Te)

            sim = sim.masked_fill(mask.unsqueeze(0) == 0, -1e9)

            attn = torch.softmax(sim, dim=-1)
            attn = attn * mask.unsqueeze(0)
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-6)

        else:
            attn = torch.softmax(sim, dim=-1)

        # 🔥 5. aligned time feature
        aligned_time = torch.einsum("bts,bsd->btd", attn, V)  # (B, T, D)

        # 🔥 6. spatial broadcast
        aligned = aligned_time.unsqueeze(2).expand(-1, -1, S, -1)  # (B, T, S, D)
        aligned = aligned.reshape(B, Nv, D)

        return aligned