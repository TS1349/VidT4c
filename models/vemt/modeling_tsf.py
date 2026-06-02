"""
TimeSformer (TSF) wrapped to match the AdaMAE/VideoMAE VisionTransformer
interface used inside vemt.py.

Why a wrapper:
  - vemt.py expects video_model with forward(x, return_feat=False, return_tokens=False),
    forward_features_until(x, end_idx), forward_features_from(h, start_idx),
    plus .blocks / .norm / .head attributes for freeze and cache logic.
  - The original models/tsf/vit.py was written stand-alone with a custom block
    signature (`blk(x, B, T, W)` for divided spacetime attention) and a single
    forward_features that runs everything end to end. This wrapper splits it
    into the until/from helpers so VEMT's dense-cache path (cache block-N
    intermediate, train only the last N blocks) works for TSF too.

Notes:
  - TSF blocks need (B, T, W) shape state; we keep T = num_frames (e.g. 32)
    and W = img_size // patch_size (e.g. 14) as instance constants, since
    they're fixed for a given input size (all dense clips share the shape).
  - TSF uses CLS-token pooling (no fc_norm). No V/A split applied internally;
    the caller (vemt.py) handles K-clip / dense pool then optional V/A split,
    matching AdaMAE / VideoMAE convention.
  - Pretrained weights (tsf.pth) load directly into self.model in __init__
    if --pretrained is set.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange

from ..tsf.vit import VisionTransformer as TSFInnerVisionTransformer


class TSFVisionTransformer(nn.Module):
    def __init__(self, args,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 output_dim=None,
                 all_frames=32,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 **kwargs):
        super().__init__()
        self.args = args
        self.num_classes = output_dim[0] * output_dim[1]
        self.num_value = output_dim[0]
        self.embed_dim = embed_dim
        self.all_frames = all_frames

        self.model = TSFInnerVisionTransformer(
            img_size=img_size,
            output_dim=self.num_classes,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            num_frames=all_frames,
            attention_type='divided_space_time',
        )

        # Fixed shape state for TSF blocks (consistent across all clips of same input size).
        self._T_fixed = all_frames
        self._W_fixed = img_size // patch_size

        # Identity dropout so vemt._tail_fwd's `dropout(global_f)` is a no-op for TSF.
        self.fc_dropout = nn.Identity()

        # Load TSF Kinetics pretrained if requested. Mirrors the standalone
        # runner.py loading for --model tsf so we don't need a separate branch.
        if getattr(args, 'pretrained', False) and not getattr(args, 'set_eeg_only', False):
            ckpt_path = os.path.join(os.getcwd(), 'pretrained', 'tsf.pth')
            if os.path.exists(ckpt_path):
                try:
                    checkpoint = torch.load(ckpt_path, map_location='cpu')
                    missing, unexpected = self.model.load_state_dict(checkpoint, strict=False)
                    print(f'[TSF] loaded pretrained from {ckpt_path}. '
                          f'missing={len(missing)} unexpected={len(unexpected)}')
                except Exception as e:
                    print(f'[TSF][WARN] failed to load {ckpt_path}: {e}')
            else:
                print(f'[TSF][WARN] pretrained checkpoint not found at {ckpt_path}')

    # ---------- Properties for VEMT freeze / cache logic ----------
    @property
    def blocks(self):
        return self.model.blocks

    @property
    def norm(self):
        return self.model.norm

    @property
    def fc_norm(self):
        return None  # TSF uses CLS pool; no separate fc_norm

    @property
    def head(self):
        return self.model.head

    # ---------- Internal: embeddings + time embedding (pre-blocks) ----------
    def _embed(self, x):
        """Run patch_embed + cls concat + pos_embed + time embedding.

        Returns (h, B, T, W) — h is the token sequence ready to feed into blocks,
        T and W are the time / spatial token counts needed by divided attention.
        """
        B = x.shape[0]
        h, T, W = self.model.patch_embed(x)  # h: [B*T, num_spatial, D]
        cls_tokens = self.model.cls_token.expand(h.size(0), -1, -1)
        h = torch.cat((cls_tokens, h), dim=1)

        # pos_embed: resize if shape mismatches (e.g. when num_patches differs)
        if h.size(1) != self.model.pos_embed.size(1):
            pos_embed = self.model.pos_embed
            cls_pos_embed = pos_embed[0, 0, :].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0, 1:, :].unsqueeze(0).transpose(1, 2)
            P = int(other_pos_embed.size(2) ** 0.5)
            H = h.size(1) // W
            other_pos_embed = other_pos_embed.reshape(1, h.size(2), P, P)
            new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='nearest')
            new_pos_embed = new_pos_embed.flatten(2).transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            h = h + new_pos_embed
        else:
            h = h + self.model.pos_embed
        h = self.model.pos_drop(h)

        # Time embedding for divided attention
        if self.model.attention_type != 'space_only':
            cls_tokens = h[:B, 0, :].unsqueeze(1)
            h = h[:, 1:]
            h = rearrange(h, '(b t) n m -> (b n) t m', b=B, t=T)
            if T != self.model.time_embed.size(1):
                time_embed = self.model.time_embed.transpose(1, 2)
                new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
                h = h + new_time_embed.transpose(1, 2)
            else:
                h = h + self.model.time_embed
            h = self.model.time_drop(h)
            h = rearrange(h, '(b n) t m -> b (n t) m', b=B, t=T)
            h = torch.cat((cls_tokens, h), dim=1)
        return h, B, T, W

    # ---------- Public: cache split helpers ----------
    def forward_features_until(self, x, end_block_idx):
        """Run embeddings + blocks[0:end_block_idx]. Returns [B, 1+N, D]."""
        h, B, T, W = self._embed(x)
        for blk in self.model.blocks[:end_block_idx]:
            h = blk(h, B, T, W)
        return h

    def forward_features_from(self, h, start_block_idx):
        """Resume from blocks[start_block_idx] → norm → CLS pool. Returns [B, D].

        Uses cached self._T_fixed / self._W_fixed for the block (T, W) state
        since shape is fixed for the input size.
        """
        B = h.shape[0]
        T = self._T_fixed
        W = self._W_fixed
        for blk in self.model.blocks[start_block_idx:]:
            h = blk(h, B, T, W)
        if self.model.attention_type == 'space_only':
            h = rearrange(h, '(b t) n m -> b t n m', b=B, t=T)
            h = torch.mean(h, 1)
        h = self.model.norm(h)
        return h[:, 0]  # CLS pool

    # ---------- VEMT-compatible forward ----------
    def forward(self, x, eeg_feat=None, return_feat=False, return_tokens=False):
        """Standard forward. x: [B, C, T, H, W] (AdaMAE/VEMT layout).

        return_tokens currently not supported by TSF wrapper (CLS-only pooling);
        falls back to global feature only when requested.
        """
        h, B, T, W = self._embed(x)
        for blk in self.model.blocks:
            h = blk(h, B, T, W)
        if self.model.attention_type == 'space_only':
            h = rearrange(h, '(b t) n m -> b t n m', b=B, t=T)
            h = torch.mean(h, 1)
        h = self.model.norm(h)
        video_f = h[:, 0]  # CLS

        if return_tokens:
            tokens = h[:, 1:]  # patch tokens
            if self.args.set_video_only or self.args.fusion == 'router':
                out = self.head(self.fc_dropout(video_f))
                if (getattr(self.args, 'num_clips', 1) == 1
                        and not getattr(self.args, 'dense_video_clips', False)
                        and self.args.dataset in ('emognition', 'mdmer')):
                    x_v = out[:, :self.num_value].unsqueeze(-1)
                    x_a = out[:, self.num_value:].unsqueeze(-1)
                    out = torch.cat((x_v, x_a), dim=-1)
                if return_feat:
                    return out, video_f, tokens
                return out, tokens
            if return_feat:
                return video_f, video_f, tokens
            return video_f, tokens

        if self.args.set_video_only or self.args.fusion == 'router':
            out = self.head(self.fc_dropout(video_f))
            if (getattr(self.args, 'num_clips', 1) == 1
                    and not getattr(self.args, 'dense_video_clips', False)
                    and self.args.dataset in ('emognition', 'mdmer')):
                x_v = out[:, :self.num_value].unsqueeze(-1)
                x_a = out[:, self.num_value:].unsqueeze(-1)
                out = torch.cat((x_v, x_a), dim=-1)
            if return_feat:
                return out, video_f
            return out

        # No head path (eeg_signal + non-router fusion): video_f as logit placeholder
        if return_feat:
            return video_f, video_f
        return video_f
