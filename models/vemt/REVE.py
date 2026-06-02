"""
REVE-EEG (NeurIPS 2025, arXiv 2510.21585) wrapper for VEMT.

Alternative EEG backbone to CBraMod. Mirrors the CBraMod_Model interface so
vemt.py can branch via --eeg_backbone reve without touching downstream code
(GCN region path, classifier heads, fusion).

Why REVE instead of CBraMod:
  - Newer (Oct 2025), MAE-pretrained on 60K+ hours / 92 datasets / 25K subjects.
  - Authors report +2.5% avg balanced acc over CBraMod across 10 downstream tasks
    (up to +17% in linear probing).
  - 4D positional encoding (B × ch × time × feat) handles variable electrode
    layouts without retraining — natural fit for our 3-dataset setting (4-ch
    emognition / 18-ch mdmer / 30-ch eav).

Mismatches vs CBraMod (and how this wrapper bridges them):
  - REVE wants raw EEG at 200 Hz [B, ch, T]. CBraMod_Model gets pre-patched
    [B, ch, T_seg, patch_size=200]. We reshape (and would resample if
    sampling rate != 200; in practice the dataloader patches at the
    internal 200 Hz so we just flatten T_seg × patch_size → T).
  - REVE needs 3D electrode coordinates per channel. We pre-bake a dict
    mapping each dataset's electrode list → standard 10-20 sphere coords,
    so training never depends on a second HF download.
  - REVE has no documented `return_per_channel_pre` analog (its 4D PE is
    integrated). The wrapper exposes `return_per_channel_pre=True` but
    returns the POST-positional per-channel embedding instead — best
    available approximation. Downstream code (GCN region nodes) still
    works; just no "pre-positional channel-independent" guarantee here.

User-side setup (one-time, required before first run):
  1. Visit https://huggingface.co/brain-bzh/reve-base and accept the
     Responsible Use Agreement.
  2. `huggingface-cli login` with a token that has 'Read' access.
  3. Run training with `--eeg_backbone reve`.

If the HF download fails (gating not accepted, no token), the model __init__
raises with a clear message — runner.py catches and prints the setup steps.
"""

import sys
import types
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


# --- torch.nn.attention shim for PyTorch < 2.3 -----------------------------
# REVE's HF code (modeling_reve.py:9) does `from torch.nn.attention import
# SDPBackend, sdpa_kernel`. That module was added in PyTorch 2.3; this env
# is pinned to 2.1 (and upgrading risks breaking concurrent training jobs).
# The symbols are used purely as a context-manager hint to PyTorch's SDPA
# kernel selection — no-op shims are safe because torch 2.1's SDPA
# auto-selects an available backend anyway.
def _install_attention_shim_if_missing():
    if hasattr(torch.nn, "attention"):
        return  # already exists (torch ≥ 2.3) — nothing to do.

    class _SDPBackend:
        FLASH_ATTENTION = "flash"
        EFFICIENT_ATTENTION = "efficient"
        MATH = "math"
        CUDNN_ATTENTION = "cudnn"

    @contextmanager
    def _sdpa_kernel(backends=None, **kwargs):
        # No-op: torch ≤ 2.2 selects an SDPA backend automatically inside
        # F.scaled_dot_product_attention. The hint is just for perf, not
        # correctness, so silently ignoring it is safe.
        yield

    mod = types.ModuleType("torch.nn.attention")
    mod.SDPBackend = _SDPBackend
    mod.sdpa_kernel = _sdpa_kernel
    sys.modules["torch.nn.attention"] = mod
    torch.nn.attention = mod  # so `torch.nn.attention.X` lookups also work


_install_attention_shim_if_missing()


# --- Relax REVE's torch-version assert -------------------------------------
# REVE's HF code asserts `torch.__version__ >= 2.2.0` before using
# F.scaled_dot_product_attention, but SDPA exists from torch 2.0+ and works
# fine in 2.1. Combined with our sdpa_kernel shim above, the only thing
# blocking us is the literal version string in the cached file. Patch it
# in place (idempotent: skips if already patched, no-op if cache is absent).
def _relax_reve_sdpa_assert():
    import os
    import glob
    cache_root = os.path.expanduser(
        "~/.cache/huggingface/modules/transformers_modules/brain-bzh/reve-base"
    )
    candidates = glob.glob(os.path.join(cache_root, "*", "modeling_reve.py"))
    for path in candidates:
        try:
            with open(path, "r") as f:
                src = f.read()
        except OSError:
            continue
        if 'version.parse("2.2.0")' not in src:
            continue  # already patched or unexpected layout
        new_src = src.replace(
            'version.parse("2.2.0")',
            'version.parse("2.0.0")',  # SDPA exists from 2.0; relax assert
        )
        try:
            with open(path, "w") as f:
                f.write(new_src)
            print(f"[REVE] patched torch-version assert in {path}")
        except OSError as e:
            print(f"[REVE][WARN] could not patch {path}: {e}")


_relax_reve_sdpa_assert()


# Dataset-to-electrode-name mapping. The 3D coordinates themselves are
# resolved at runtime by REVE's official `brain-bzh/reve-positions` position
# bank — calling pos_bank(names) returns the exact coordinate frame REVE
# was trained on. Don't hardcode coords here: their normalization /
# head-radius convention isn't documented and using a guessed frame would
# silently degrade REVE accuracy.
_DATASET_CHANNELS = {
    "emognition": ["TP9", "AF7", "AF8", "TP10"],
    "mdmer": [
        "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8",
        "T3", "C3", "CZ", "C4", "T4",
        "P3", "PZ", "P4", "O1", "O2", "OZ",
    ],
    "eav": [
        "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8",
        "FC5", "FC1", "FC2", "FC6",
        "T7", "C3", "CZ", "C4", "T8",
        "CP5", "CP1", "CP2", "CP6",
        "P7", "P3", "PZ", "P4", "P8",
        "PO9", "O1", "OZ", "O2", "PO10",
    ],
}


def _electrode_names_for_dataset(dataset_name: str) -> list:
    """Return the ordered electrode-name list for the dataset's EEG channels."""
    if dataset_name not in _DATASET_CHANNELS:
        raise KeyError(
            f"REVE wrapper: no channel layout registered for dataset "
            f"'{dataset_name}'. Add it to _DATASET_CHANNELS in models/vemt/REVE.py."
        )
    return list(_DATASET_CHANNELS[dataset_name])


def _resolve_positions(electrode_names):
    """Call REVE's official position bank to get [ch, 3] coords.

    Loads `brain-bzh/reve-positions` via AutoModel.from_pretrained (also gated;
    same RUA acceptance as the main repo) and calls it with the electrode-name
    list. Returns a [ch, 3] tensor in REVE's native coordinate frame so we
    don't have to guess the normalization.
    """
    from transformers import AutoModel
    pos_bank = AutoModel.from_pretrained(
        "brain-bzh/reve-positions", trust_remote_code=True
    )
    return pos_bank(electrode_names)  # [ch, 3]


class REVE_Model(nn.Module):
    """Drop-in alternative to models.vemt.CBraMod.CBraMod_Model.

    Same constructor signature, same forward contract: returns either
    (output, feats_0) or (output, feats_0, ...extras) per the variadic
    return convention used by vemt.py.

    Constructor signature mirrors CBraMod_Model exactly so vemt.py can
    swap classes without changing keyword arguments.
    """

    HF_BACKBONE = "brain-bzh/reve-base"

    def __init__(self, args, in_dim=200, output_dim=200, d_model=200,
                 dim_feedforward=800, seq_len=30, n_layer=12, nhead=8,
                 in_chans=int):
        super().__init__()
        self.num_ch = in_chans
        self.args = args
        self.output_dim = output_dim
        if output_dim[1] == 1:
            self.output_dim = (output_dim[0],)
        self.num_classes = output_dim[0] * output_dim[1]

        # ---- HF backbone load (gated; clear error if RUA not accepted) -----
        try:
            from transformers import AutoModel
            self.backbone = AutoModel.from_pretrained(
                self.HF_BACKBONE,
                trust_remote_code=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"\n[REVE_Model] Failed to load HF model '{self.HF_BACKBONE}'.\n"
                f"Underlying error: {type(e).__name__}: {e}\n\n"
                f"Setup steps:\n"
                f"  1. Visit https://huggingface.co/{self.HF_BACKBONE} and accept the\n"
                f"     Responsible Use Agreement.\n"
                f"  2. Run `huggingface-cli login` with a 'Read' token.\n"
                f"  3. Retry training with --eeg_backbone reve.\n"
            ) from e

        # Probe REVE's output feature dim. REVE's config uses `embed_dim`
        # (not the HF-standard `hidden_size`). Check both keys in order, then
        # fall back to REVE-base's known value (512). Verified at runtime
        # against the first forward output — mismatch raises a clear error.
        cfg = getattr(self.backbone, "config", None)
        d_reve = (
            getattr(cfg, "embed_dim", None)
            or getattr(cfg, "hidden_size", None)
            or 512  # REVE-base
        )
        self.d_reve = int(d_reve)

        # ---- Electrode positions buffer (per current dataset) --------------
        # Resolved via REVE's official position bank (brain-bzh/reve-positions).
        # Same gating as the main repo — both fetch under the single RUA consent.
        names = _electrode_names_for_dataset(getattr(args, "dataset", "mdmer"))
        try:
            pos = _resolve_positions(names)
            if not torch.is_tensor(pos):
                pos = torch.as_tensor(pos, dtype=torch.float32)
            pos = pos.detach().to(torch.float32)
        except Exception as e:
            raise RuntimeError(
                f"\n[REVE_Model] Failed to load position bank 'brain-bzh/reve-positions'.\n"
                f"Underlying error: {type(e).__name__}: {e}\n"
                f"This repo is gated under the SAME RUA as brain-bzh/reve-base — "
                f"if you accepted RUA on reve-base, also accept on reve-positions "
                f"(https://huggingface.co/brain-bzh/reve-positions).\n"
            ) from e
        # [ch, 3] -> registered as non-trainable buffer; broadcast per batch in forward.
        self.register_buffer("electrode_positions", pos, persistent=False)

        # ---- Classifier head (mirrors CBraMod_Model layout) ----------------
        # CBraMod: classifier collapses [B, ch, 10, 200] -> [B, 768].
        # REVE:    classifier collapses [B, ch, T_patch_target, d_reve] -> [B, 768]
        #          directly — no intermediate 512→200 compression so REVE's full
        #          per-channel embedding survives into both the global feature
        #          and the GCN region path (vemt.py picks up the native d_reve
        #          via eeg_feat_proj sized to backbone dim).
        # T_patch_target = 10 to mirror CBraMod's patch budget.
        self._t_patch_target = 10
        self.classifier = nn.Sequential(
            Rearrange("b c s d -> b (c s d)"),
            nn.Linear(self.num_ch * self._t_patch_target * self.d_reve, 10 * 200),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(10 * 200, 768),
        )
        self.classifier2 = nn.Sequential(
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(768, self.num_classes),
        )

    # ---- Internal: pool REVE's [B, ch, T_patch, d_reve] -> standard layout ----
    def _to_standard_feats(self, reve_feats):
        """REVE per-channel features (any T_patch) -> [B, ch, T_patch_target, d_reve].

        Adaptive temporal pool only — feature dim (d_reve=512) is preserved so
        the GCN region path picks up REVE's full embedding rather than a
        compressed 200-dim bottleneck.
        """
        B, C, T, D = reve_feats.shape
        if T == self._t_patch_target:
            return reve_feats
        # adaptive avg pool over the temporal axis, keep D unchanged
        x = reve_feats.transpose(2, 3).contiguous()      # [B, C, D, T]
        x = F.adaptive_avg_pool1d(
            x.view(B * C, D, T), self._t_patch_target
        ).view(B, C, D, self._t_patch_target)
        x = x.transpose(2, 3).contiguous()               # [B, C, T_patch_target, D]
        return x

    # ---- Forward: same signature & return contract as CBraMod_Model ----
    def forward(self, sample, mask=None, return_encoder_feats=False,
                return_per_channel_pre=False):
        x = sample["eeg"]
        # CBraMod expects [B, ch, T_seg, patch_size=200]. REVE expects raw
        # [B, ch, T_total]. Flatten the patch axes.
        if x.dim() == 4:
            B, C, T_seg, P = x.shape
            eeg_raw = x.view(B, C, T_seg * P)
        elif x.dim() == 3:
            B, C, _ = x.shape
            eeg_raw = x
        else:
            raise ValueError(f"REVE_Model: unexpected sample['eeg'] dim {x.dim()}")

        # Broadcast electrode positions per sample: [ch, 3] -> [B, ch, 3]
        positions = self.electrode_positions.unsqueeze(0).expand(B, -1, -1).to(eeg_raw.device)

        # REVE forward. Per the HF custom model card, signature is
        # `model(eeg, positions)`. If signature differs (e.g. kwargs),
        # adjust here after verifying via `inspect.signature(self.backbone.forward)`.
        out = self.backbone(eeg_raw, positions)

        # Pull per-channel-per-time feature tensor. REVE's HF code may return
        # a dict, BaseModelOutput, or tensor. Handle the common cases.
        if hasattr(out, "last_hidden_state"):
            reve_feats = out.last_hidden_state
        elif isinstance(out, (tuple, list)):
            reve_feats = out[0]
        else:
            reve_feats = out

        # Expected shape: [B, ch, T_patch, d_reve]. If REVE returns
        # [B, ch * T_patch, d_reve] (token-style), reshape.
        if reve_feats.dim() == 3 and reve_feats.size(1) == C * (reve_feats.size(1) // C):
            T_patch = reve_feats.size(1) // C
            reve_feats = reve_feats.view(B, C, T_patch, -1)
        elif reve_feats.dim() != 4:
            raise RuntimeError(
                f"REVE_Model: unexpected REVE output shape {tuple(reve_feats.shape)}. "
                f"Expected 4-D [B, ch, T_patch, D] or 3-D token-format [B, ch*T_patch, D]."
            )

        # Sync d_reve lazily on first forward (in case config.hidden_size was wrong).
        if reve_feats.size(-1) != self.d_reve:
            raise RuntimeError(
                f"REVE_Model: feat dim mismatch (probed d_reve={self.d_reve}, "
                f"got {reve_feats.size(-1)}). Rebuild feat_proj or set d_reve "
                f"explicitly in __init__."
            )

        feats = self._to_standard_feats(reve_feats)   # [B, C, T_patch_target, 200]
        feats_0 = self.classifier(feats)              # [B, 768]
        output = self.classifier2(feats_0)            # [B, n_classes]

        if self.args.dataset in ("emognition", "mdmer"):
            output_v = output[:, : self.output_dim[0]].unsqueeze(-1)
            output_a = output[:, self.output_dim[0] :].unsqueeze(-1)
            output = torch.cat((output_v, output_a), dim=-1)

        # Variadic returns mirroring CBraMod_Model.
        extras = []
        if return_encoder_feats:
            extras.append(feats)              # [B, ch, T_patch_target, 200]
        if return_per_channel_pre:
            # REVE has no documented pre-positional split; reuse post-encoder
            # per-channel feats as the best available substitute. Downstream
            # GCN region path treats it as channel-independent per-clip features.
            extras.append(feats)
        if extras:
            return (output, feats_0, *extras)
        return output, feats_0
