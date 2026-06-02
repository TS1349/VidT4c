"""
HuggingFace ViViT wrapped to match the AdaMAE/VideoMAE VisionTransformer
interface used inside vemt.py.

Why a wrapper:
  vemt.py expects the video backbone to:
    - Accept [B, C, T, H, W] input (AdaMAE convention; vemt.py transposes
      x["video"] from [B, T, C, H, W] to [B, C, T, H, W] before calling).
    - forward(x, eeg_feat=None, return_feat=False, return_tokens=False)
      returning logit / (logit, feat) / (logit, feat, tokens) per the same
      contract as modeling_finetune_v0.VisionTransformer.
    - Expose .blocks (ModuleList), .head, and a post-block norm so
      freeze_backbones can keep the last N blocks trainable.

HF ViViT layout we tap into:
  model.vivit.embeddings.patch_embeddings
  model.vivit.encoder.layer       (ModuleList[12])
  model.vivit.layernorm           (final LayerNorm before classifier)
  model.classifier                (Linear)
  output.last_hidden_state shape: [B, 1+T'*S, D]  (CLS at index 0)
"""

import torch
import torch.nn as nn
from transformers import VivitConfig, VivitForVideoClassification


class VivitVisionTransformer(nn.Module):
    def __init__(self, args,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 output_dim=None,
                 all_frames=32,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 tubelet_size=2,
                 **kwargs):
        super().__init__()
        self.args = args
        self.num_classes = output_dim[0] * output_dim[1]
        self.num_value = output_dim[0]
        self.embed_dim = embed_dim
        self.all_frames = all_frames

        vivit_config = VivitConfig(
            image_size=img_size,
            num_frames=all_frames,
            tubelet_size=[tubelet_size, patch_size, patch_size],
            video_size=[all_frames, img_size, img_size],
            num_channels=in_chans,
            hidden_size=embed_dim,
            num_hidden_layers=depth,
            num_attention_heads=num_heads,
            num_labels=self.num_classes,
        )

        if getattr(args, 'pretrained', False):
            self.model = VivitForVideoClassification.from_pretrained(
                "google/vivit-b-16x2-kinetics400",
                config=vivit_config,
                ignore_mismatched_sizes=True,
                use_safetensors=True,
            )
        else:
            self.model = VivitForVideoClassification(config=vivit_config)

        self.fc_dropout = nn.Identity()

    @property
    def blocks(self):
        return self.model.vivit.encoder.layer

    @property
    def norm(self):
        return self.model.vivit.layernorm

    @property
    def fc_norm(self):
        return None

    @property
    def head(self):
        return self.model.classifier

    def forward_features_until(self, x, end_block_idx):
        """Run embeddings + encoder layers [0, end_block_idx). Returns [B, 1+N, D].
        Mirrors AdaMAE / VideoMAE helper so VEMT.dense_cache_features works.
        """
        # vemt.py feeds [B, C, T, H, W]; HF ViViT wants [B, T, C, H, W]
        x = x.transpose(-3, -4).contiguous()
        h = self.model.vivit.embeddings(x)  # [B, 1+N, D]
        for blk in self.model.vivit.encoder.layer[:end_block_idx]:
            h = blk(h)[0]
        return h

    def forward_features_from(self, h, start_block_idx):
        """Resume from encoder.layer[start_block_idx] → layernorm → CLS pool.
        Returns global feature [B, D].
        """
        for blk in self.model.vivit.encoder.layer[start_block_idx:]:
            h = blk(h)[0]
        h = self.model.vivit.layernorm(h)
        return h[:, 0]  # CLS token

    def forward(self, x, eeg_feat=None, return_feat=False, return_tokens=False):
        # vemt.py feeds [B, C, T, H, W] (AdaMAE layout); HF ViViT wants [B, T, C, H, W]
        x = x.transpose(-3, -4).contiguous()

        outputs = self.model.vivit(x, return_dict=True)
        last_hidden = outputs.last_hidden_state          # [B, 1+N, D], norm already applied
        global_f = last_hidden[:, 0]                     # CLS token  [B, D]
        tokens = last_hidden[:, 1:]                      # patch tokens [B, N, D]

        if return_tokens:
            if self.args.set_video_only or self.args.fusion == 'router':
                out = self.head(global_f)
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

        if self.args.set_video_only or self.args.fusion == 'router':
            out = self.head(global_f)
            if self.args.num_clips == 1 and self.args.dataset in ('emognition', 'mdmer'):
                x_v = out[:, :self.num_value].unsqueeze(-1)
                x_a = out[:, self.num_value:].unsqueeze(-1)
                out = torch.cat((x_v, x_a), dim=-1)
            if return_feat:
                return out, global_f
            return out

        if return_feat:
            return global_f, global_f
        return global_f
