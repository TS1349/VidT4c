"""
Video Swin Transformer (torchvision swin3d_b) wrapped to match the
AdaMAE / VideoMAE / ViViT / TSF VisionTransformer interface used inside vemt.py.

Why a wrapper:
  vemt.py expects video_model to:
    - accept [B, C, T, H, W] (AdaMAE convention; vemt.py transposes
      x["video"] from [B, T, C, H, W] to [B, C, T, H, W] before calling),
    - forward(x, eeg_feat=None, return_feat=False, return_tokens=False)
      with the same logit / (logit, feat) / (logit, feat, tokens) contract
      as modeling_finetune_v0.VisionTransformer,
    - expose .blocks / .norm / .head so freeze_backbones and the dense
      cache split (cache pre-trainable, train only last N blocks) work.

Dimension note:
  swin3d_b's final feature dim is 1024, while VEMT downstream modules
  (GCN region/local, eeg_feat_proj, clip_attn_v, ...) are hard-wired to
  self.embed_dim = 768. We add a 1024->768 projection INSIDE the wrapper
  so callers see a 768-dim feature and don't have to special-case Swin.
  The classifier head then maps 768 -> num_classes.

Blocks semantic for freeze / cache:
  We expose model.features (length 7: 4 SwinStage Sequentials interleaved
  with 3 PatchMerging) as .blocks. --video_unfreeze_last_n_blocks N keeps
  the last N entries of model.features trainable. N=1 = only the final
  Sequential (last 2 transformer blocks). The cache split happens at the
  same boundary, so forward_features_until / forward_features_from
  compose cleanly.

Pretrained:
  We load Swin3D_B_Weights.KINETICS400_V1 directly via torchvision (with
  the original 400-class head) and then swap in our projection + new head.
  Mirrors the standalone BridgedVideoSwin4C loading path so we don't need
  a separate branch in runner.py.
"""

import torch
import torch.nn as nn

from torchvision.models.video.swin_transformer import swin3d_b, Swin3D_B_Weights


class SwinVisionTransformer(nn.Module):
    def __init__(self, args,
                 img_size=224,
                 patch_size=16,           # accepted for API parity; Swin uses internal 4x4 patches
                 in_chans=3,
                 output_dim=None,
                 all_frames=32,           # accepted for API parity
                 embed_dim=768,           # downstream-facing dim
                 depth=12,                # ignored; Swin depth is fixed by variant
                 num_heads=12,            # ignored
                 tubelet_size=2,          # ignored; Swin uses internal 2x4x4 patches
                 **kwargs):
        super().__init__()
        self.args = args
        self.num_classes = output_dim[0] * output_dim[1]
        self.num_value = output_dim[0]
        self.embed_dim = embed_dim
        self.all_frames = all_frames

        weights = (Swin3D_B_Weights.KINETICS400_V1
                   if getattr(args, 'pretrained', False) and not getattr(args, 'set_eeg_only', False)
                   else None)
        # Load with original 400-class head so pretrained weights match,
        # then replace head with our (in_dim -> embed_dim -> num_classes) pipeline.
        self.model = swin3d_b(weights=weights, num_classes=400)
        backbone_dim = self.model.head.in_features  # 1024 for swin3d_b
        self.model.head = nn.Identity()             # we own the head externally

        # 1024 -> 768 projection so downstream VEMT modules see embed_dim.
        # Identity if the backbone already matches embed_dim.
        if backbone_dim == embed_dim:
            self.feat_proj = nn.Identity()
        else:
            self.feat_proj = nn.Linear(backbone_dim, embed_dim)

        # Classifier head (set_video_only / router fusion). Trainable.
        self._head = nn.Linear(embed_dim, self.num_classes)

        # Identity dropout so vemt._tail_fwd's dropout(global_f) is a no-op.
        self.fc_dropout = nn.Identity()

    # ---------- Properties for VEMT freeze / cache logic ----------
    @property
    def blocks(self):
        # 7 stages: SwinStage / PatchMerging / SwinStage / PatchMerging / SwinStage / PatchMerging / SwinStage
        return self.model.features

    @property
    def norm(self):
        return self.model.norm

    @property
    def fc_norm(self):
        return None

    @property
    def head(self):
        return self._head

    # ---------- Internal: shared head over pooled backbone feature ----------
    def _pool_and_project(self, h):
        """h: [B, T', H', W', C_backbone] (post-norm). Returns global feat [B, embed_dim]."""
        h = h.permute(0, 4, 1, 2, 3)              # [B, C, T', H', W']
        h = self.model.avgpool(h)                 # [B, C, 1, 1, 1]
        h = torch.flatten(h, 1)                   # [B, C_backbone]
        return self.feat_proj(h)                  # [B, embed_dim]

    # ---------- Public: cache split helpers ----------
    def forward_features_until(self, x, end_block_idx):
        """Run patch_embed + pos_drop + features[0:end_block_idx]. Returns post-stage map."""
        h = self.model.patch_embed(x)
        h = self.model.pos_drop(h)
        for stage in self.model.features[:end_block_idx]:
            h = stage(h)
        return h

    def forward_features_from(self, h, start_block_idx):
        """Resume features[start_block_idx:] -> norm -> pool -> project. Returns [B, embed_dim]."""
        for stage in self.model.features[start_block_idx:]:
            h = stage(h)
        h = self.model.norm(h)
        return self._pool_and_project(h)

    # ---------- VEMT-compatible forward ----------
    def forward(self, x, eeg_feat=None, return_feat=False, return_tokens=False):
        """x: [B, C, T, H, W] (AdaMAE/VEMT layout)."""
        h = self.model.patch_embed(x)
        h = self.model.pos_drop(h)
        for stage in self.model.features:
            h = stage(h)
        h = self.model.norm(h)                        # [B, T', H', W', C_backbone]

        if return_tokens:
            B, T1, H1, W1, Cb = h.shape
            tokens_backbone = h.reshape(B, T1 * H1 * W1, Cb)
            tokens = self.feat_proj(tokens_backbone)  # [B, T'*H'*W', embed_dim]
        global_f = self._pool_and_project(h)          # [B, embed_dim]

        # Decide whether to apply head + V/A split (mirrors ViViT/TSF wrapper logic).
        use_head = self.args.set_video_only or self.args.fusion == 'router'
        single_clip_va = (
            getattr(self.args, 'num_clips', 1) == 1
            and not getattr(self.args, 'dense_video_clips', False)
            and self.args.dataset in ('emognition', 'mdmer')
        )

        if return_tokens:
            if use_head:
                out = self._head(self.fc_dropout(global_f))
                if single_clip_va:
                    x_v = out[:, :self.num_value].unsqueeze(-1)
                    x_a = out[:, self.num_value:].unsqueeze(-1)
                    out = torch.cat((x_v, x_a), dim=-1)
                return (out, global_f, tokens) if return_feat else (out, tokens)
            return (global_f, global_f, tokens) if return_feat else (global_f, tokens)

        if use_head:
            out = self._head(self.fc_dropout(global_f))
            if single_clip_va:
                x_v = out[:, :self.num_value].unsqueeze(-1)
                x_a = out[:, self.num_value:].unsqueeze(-1)
                out = torch.cat((x_v, x_a), dim=-1)
            return (out, global_f) if return_feat else out

        # Fusion (gcn/film/...) path: caller uses global_f, head is unused.
        return (global_f, global_f) if return_feat else global_f
