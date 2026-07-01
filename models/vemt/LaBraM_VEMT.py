"""LaBraM EEG backbone adapted to the VEMT fusion EEG-branch contract.

Mirrors CBraMod_Model so VEMT can use LaBraM as its EEG node producer
(`--eeg_backbone labram`), exactly like the REVE wrapper does for REVE-EEG.

Contract (same as models/vemt/CBraMod.py:CBraMod_Model.forward):
  forward(sample, mask=None, return_encoder_feats=False, return_per_channel_pre=False)
    -> (output, feats_0)                                   # base
    -> (output, feats_0, encoder_feats[, per_channel_pre]) # with extras
  where
    output    : [B, num_classes]  (reshaped to [B, C, 2] for emognition/mdmer)
    feats_0   : [B, 768]  pooled EEG node feature for the GCN (matches CBraMod)
    enc_feats : [B, ch, patch_num, D]  per-(channel,patch) tokens

Input `sample["eeg"]` is the CBraMod-format tensor [B, ch, patch_num, 200] produced
by the default `--fft_mode cbramod` preprocessing — LaBraM consumes the identical
layout, so no extra EEG preprocessing is needed.

Pretrained weights (pretrained/labram.pth, 'student.' prefix) are loaded by
runner.py into `self.labram` via load_flexible, same as the standalone `--model
labram` path.
"""
import torch
import torch.nn as nn

from models.EEGs.LaBraM import LaBraM_Model


class LaBraM_VEMT_Model(nn.Module):
    def __init__(self, args, output_dim, in_chans, embed_dim=200):
        super().__init__()
        self.args = args
        self.output_dim = output_dim
        if output_dim[1] == 1:
            self.output_dim = (output_dim[0],)
        self.num_classes = output_dim[0] * output_dim[1]

        # Underlying LaBraM backbone (pretrained loaded later by runner.py).
        self.labram = LaBraM_Model(args, output_dim=output_dim, in_chans=in_chans,
                                   embed_dim=embed_dim)
        # Drop LaBraM submodules unused on the fusion path so DDP (num_gpus>1) does
        # not flag them as unused parameters:
        #   - .head: we attach our own classifier2 below.
        #   - .eeg_reduce: never called (its use is commented out in LaBraM.forward).
        self.labram.head = nn.Identity()
        if hasattr(self.labram, "eeg_reduce"):
            del self.labram.eeg_reduce

        # CBraMod-parity head: LaBraM pooled feature (embed_dim) -> 768 node feature,
        # then a small classifier for the per-modality logit (used by aux losses / eval).
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 768),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(768, 768),
        )
        self.classifier2 = nn.Sequential(
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(768, self.num_classes),
        )

    def forward(self, sample, mask=None, return_encoder_feats=False,
                return_per_channel_pre=False):
        eeg = sample["eeg"]                       # [B, ch, patch_num, 200]
        B, ch, patch_num, _ = eeg.shape
        input_chans = self.labram.default_input_chans

        need_tokens = return_encoder_feats or return_per_channel_pre
        if need_tokens:
            tokens = self.labram.forward_features(
                eeg, input_chans=input_chans, return_patch_tokens=True)  # [B, ch*patch, D]
            pooled = tokens.mean(dim=1)                                  # [B, D]
        else:
            pooled = self.labram.forward_features(
                eeg, input_chans=input_chans, return_patch_tokens=False)  # [B, D]

        feats_0 = self.classifier(pooled)         # [B, 768]
        output = self.classifier2(feats_0)        # [B, num_classes]

        if self.args.dataset in ("emognition", "mdmer"):
            output_v = output[:, :self.output_dim[0]].unsqueeze(-1)
            output_a = output[:, self.output_dim[0]:].unsqueeze(-1)
            output = torch.cat((output_v, output_a), dim=-1)

        extras = []
        if need_tokens:
            D = tokens.size(-1)
            enc = tokens.view(B, ch, patch_num, D)   # [B, ch, patch_num, D]
            if return_encoder_feats:
                extras.append(enc)
            if return_per_channel_pre:
                extras.append(enc)
        if extras:
            return (output, feats_0, *extras)
        return output, feats_0
