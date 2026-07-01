"""
EAV (Hong et al., Nature Scientific Data 2024, doi 10.1038/s41597-024-03838-4)
baseline reimplementation for VEMT comparison.

Original repo (https://github.com/nubcico/EAV) provides two modality-separate
baselines but no V+E fusion model out of the box:
  - EEG  branch: ShallowConvNet (Schirrmeister et al. 2017) at 200 Hz.
    Input [B, 1, ch, 500] → softmax probs [B, 5].
  - Vision branch: HuggingFace ViT-B/16 image classifier finetuned per frame.
    Input [B, 3, 224, 224] → logits [B, 5].

We add a late-fusion head (concat-MLP) on top so the comparison "EAV's two
branches + late fusion" matches the most common V+E fusion strategy in the
literature. This is the closest fair adaptation given EAV publishes branches
but not a fusion architecture.

Wrapper adaptations (Phase 1 honest reproduction):
  - sample["video"] [B, T, C, H, W]: **all T frames** are passed through
    the per-frame ViT and the resulting logits are averaged. This matches
    EAV's published protocol (per-frame classification, sequence-level
    aggregation). Dense N-clip mode is still collapsed (mean over clip
    axis) before the per-frame split.
  - sample["eeg"] [B, ch, T_seg, P] (CBraMod-patched 200 Hz): goes through
    EAV's actual pipeline rather than naive resampling.
      1. Flatten patches → raw EEG [B, ch, T_seg*P] at 200 Hz.
      2. **Downsample 200 Hz → 100 Hz** by decimation (EAV's target rate;
         their pipeline starts from 500 Hz and downsamples to 100 Hz).
      3. **Bandpass 0.5–45 Hz** via FFT mask (close approximation of
         EAV's 5th-order Butterworth — FFT mask gives a sharper transition
         but identical passband; differentiable, runs on GPU).
      4. **Split into N non-overlapping 5-sec windows.** EAV's protocol
         (verified from code + Nature SciData paper Methods §EEG
         preprocessing) segments every 20-sec trial into 4 non-overlapping
         5-sec windows and treats each as an independent example for both
         training and evaluation. For our 10-sec trial → N=2 windows.
      5. **Trial-atomic adaptation:**
           - Training: pick ONE random window per sample (stochastic
             equivalent of EAV's N× data augmentation while keeping
             one-trial-one-label semantics).
           - Eval: forward ALL N windows through ShallowConvNet and average
             logits → trial-level prediction (EAV reports window-level
             accuracy on its window-atomic dataset; for trial-atomic data
             the honest mapping is to ensemble window predictions).
    Net per-window result: [B, 1, ch, 500] at 100 Hz, batched as
    [B*N_seg, ...] inside the forward and averaged back to [B, n_classes].
  - Output reshaped to [B, n_v, 2] for emognition / mdmer V/A.

Reference:
  - EAV paper:       https://www.nature.com/articles/s41597-024-03838-4
  - Repo:            https://github.com/nubcico/EAV
  - ShallowConvNet:  Schirrmeister et al., HBM 2017, arXiv:1703.05051
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


_EAV_HF_VISION = "google/vit-base-patch16-224"


class _ShallowConvNet(nn.Module):
    """Schirrmeister et al. 2017 ShallowConvNet (EEG classifier).

    Mirrors the structure used in EAV repo Transformer_torch/Transformer_EEG.py:
      - Conv2d(1, 40, (1, 25))           — temporal filtering
      - Conv2d(40, 40, (ch, 1))          — spatial mixing across electrodes
      - BatchNorm2d(40)
      - square activation
      - AvgPool2d((1, 75), stride=(1, 15))
      - log (clamped)
      - Dropout(0.5)
      - Linear -> n_classes
    """

    def __init__(self, n_classes, n_channels, n_samples=500):
        super().__init__()
        self.n_samples = n_samples
        self.temporal = nn.Conv2d(1, 40, kernel_size=(1, 25), stride=1)
        self.spatial = nn.Conv2d(40, 40, kernel_size=(n_channels, 1), stride=1)
        self.bn = nn.BatchNorm2d(40)
        self.pool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))
        self.dropout = nn.Dropout(0.5)
        # Pre-compute flatten dim via a dry forward.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            x = self.spatial(self.temporal(dummy))
            x = self.bn(x)
            x = x ** 2
            x = self.pool(x)
            x = torch.log(x.clamp(min=1e-6))
            flat = x.flatten(1).shape[-1]
        self.head = nn.Linear(flat, n_classes)

    def forward(self, x):
        """x: [B, 1, ch, n_samples] → logits [B, n_classes]"""
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.bn(x)
        x = x ** 2
        x = self.pool(x)
        x = torch.log(x.clamp(min=1e-6))
        x = self.dropout(x.flatten(1))
        return self.head(x)


class EAVModel(nn.Module):
    """EAV-style V+E classifier (ShallowConvNet EEG + ViT video + late fusion).

    Constructor signature matches VEMT.
    """

    # EAV protocol (verified against nubcico/EAV repo, June 2026):
    #   downsample raw EEG to 100 Hz, bandpass 0.5–45 Hz Butterworth,
    #   then ShallowConvNet on 500-sample (5-sec) windows.
    EEG_INPUT_RATE = 200      # our CBraMod transform produces 200 Hz patched EEG
    EEG_TARGET_RATE = 100     # EAV's downsampled rate
    EEG_TARGET_SAMPLES = 500  # 5 sec window @ 100 Hz
    EEG_BAND_LO = 0.5
    EEG_BAND_HI = 45.0

    def __init__(self, args, output_dim, image_size=224, eeg_channels=18,
                 frequency_bins=None):
        super().__init__()
        self.args = args
        self.output_dim = output_dim
        if isinstance(output_dim, tuple) and len(output_dim) > 1:
            self.num_value = output_dim[0]
            self.num_classes = output_dim[0] * output_dim[1]
        else:
            self.num_value = output_dim[0] if isinstance(output_dim, tuple) else output_dim
            self.num_classes = self.num_value

        # --- EEG branch ---------------------------------------------
        self.eeg_net = _ShallowConvNet(
            n_classes=self.num_classes,
            n_channels=eeg_channels,
            n_samples=self.EEG_TARGET_SAMPLES,
        )

        # --- Vision branch -----------------------------------------
        self.vid_model = self._build_vision_model()

        # --- Late fusion head --------------------------------------
        # Concat 2 logits → linear → final logits.
        self.fuse = nn.Sequential(
            nn.Linear(2 * self.num_classes, self.num_classes),
        )

    def _build_vision_model(self):
        try:
            from transformers import AutoModelForImageClassification
            try:
                m = AutoModelForImageClassification.from_pretrained(
                    _EAV_HF_VISION,
                    num_labels=self.num_classes,
                    ignore_mismatched_sizes=True,
                )
                print(f"[EAV] vision branch: HF {_EAV_HF_VISION}")
                return m
            except Exception as e:
                print(f"[EAV][WARN] HF ViT load failed ({type(e).__name__}: {e}). "
                      f"Using torchvision vit_b_16 (random init).")
        except ImportError:
            print("[EAV][WARN] transformers missing — falling back to torchvision vit_b_16.")
        from torchvision.models import vit_b_16
        m = vit_b_16(weights=None)
        m.heads.head = nn.Linear(m.heads.head.in_features, self.num_classes)
        return m

    def _vit_forward(self, x):
        """Unify HF and torchvision ViT into logits [B, n_classes]."""
        try:
            from transformers.modeling_utils import PreTrainedModel
            if isinstance(self.vid_model, PreTrainedModel):
                return self.vid_model(pixel_values=x).logits
        except ImportError:
            pass
        return self.vid_model(x)

    def _prepare_video_frames(self, video):
        """Reshape video for per-frame ViT, return [B*T, C, H, W] + T.

        EAV's protocol: each frame goes through ViT independently, logits
        are averaged across T to form the video-level prediction. This is
        substantially more faithful than middle-frame-only (the prior
        simplification lost temporal dynamics).

        - [B, N, T, C, H, W] (dense): mean over clip axis → [B, T, C, H, W].
        - [B, T, C, H, W]: per-frame.
        - [B, K, T, C, H, W] (K-clip): mean over K → [B, T, C, H, W].
        - [B, C, H, W]: already single-frame, T=1.
        """
        if video.dim() == 6:
            video = video.mean(dim=1)
        if video.dim() == 5:
            B, T, C, H, W = video.shape
            return video.reshape(B * T, C, H, W), B, T
        if video.dim() == 4:
            B, C, H, W = video.shape
            return video, B, 1
        raise ValueError(f"EAVModel: unexpected video shape {tuple(video.shape)}")

    @staticmethod
    def _fft_bandpass(x, low_hz, high_hz, fs):
        """FFT-based bandpass mask. Differentiable, GPU-friendly.

        Close approximation of EAV's 5th-order Butterworth bandpass: same
        passband edges, but FFT mask gives a brick-wall transition rather
        than the Butterworth's gradual rolloff. For our purpose (matching
        what ShallowConvNet sees within the passband) the difference is
        negligible — temporal filters care about the passband content,
        not the transition steepness.
        """
        N = x.size(-1)
        freqs = torch.fft.rfftfreq(N, d=1.0 / fs).to(x.device)
        mask = ((freqs >= low_hz) & (freqs <= high_hz)).to(x.dtype)
        X = torch.fft.rfft(x, dim=-1)
        X = X * mask
        return torch.fft.irfft(X, n=N, dim=-1)

    def _prepare_eeg(self, eeg):
        """[B, ch, T_seg, P] → [B, N_seg, 1, ch, 500] (training: N_seg=1, eval: N_seg=all).

        Faithful EAV pipeline (verified against nubcico/EAV code + paper):
          1. Flatten patches → raw EEG at EEG_INPUT_RATE.
          2. Decimate → EEG_TARGET_RATE.
          3. Bandpass 0.5–45 Hz (FFT mask, Butterworth-equivalent passband).
          4. Slice into N non-overlapping 500-sample windows.
          5. Training: pick ONE window at random (stochastic equivalent of
             EAV's N× per-trial data augmentation).
             Eval: keep ALL windows so forward can average their logits
             (gives trial-level prediction, since our dataloader is trial-
             atomic — EAV reports window-level accuracy on its window-atomic
             dataset, but for trial-atomic data the honest mapping is to
             ensemble window predictions).
        """
        if eeg.dim() == 5:
            eeg = eeg.mean(dim=1)
        if eeg.dim() != 4:
            raise ValueError(f"EAVModel: unexpected EEG shape {tuple(eeg.shape)}")

        # 1. Flatten patches → raw waveform.
        B, ch, T_seg, P = eeg.shape
        eeg = eeg.reshape(B, ch, T_seg * P)  # [B, ch, L] at EEG_INPUT_RATE

        # 2. Decimate.
        decim = self.EEG_INPUT_RATE // self.EEG_TARGET_RATE
        if decim > 1:
            eeg = eeg[..., ::decim]            # [B, ch, L/decim]

        # 3. Bandpass 0.5–45 Hz at EEG_TARGET_RATE.
        eeg = self._fft_bandpass(
            eeg, low_hz=self.EEG_BAND_LO, high_hz=self.EEG_BAND_HI,
            fs=self.EEG_TARGET_RATE,
        )

        # 4. Slice into N non-overlapping windows of EEG_TARGET_SAMPLES.
        L = eeg.size(-1)
        target = self.EEG_TARGET_SAMPLES
        if L < target:
            # Defensive: pad-interp up to one window's worth.
            eeg = F.interpolate(eeg, size=target, mode="linear", align_corners=False)
            return eeg.unsqueeze(1).unsqueeze(1)        # [B, 1, 1, ch, 500]

        n_seg = L // target                    # floor — drop any tail < 500
        eeg = eeg[..., :n_seg * target]
        eeg = eeg.reshape(B, ch, n_seg, target)
        eeg = eeg.permute(0, 2, 1, 3)          # [B, N_seg, ch, 500]

        # 5. Train: random 1 segment; Eval: keep all.
        if self.training and n_seg > 1:
            sel = torch.randint(0, n_seg, (B,), device=eeg.device)
            eeg = eeg[torch.arange(B, device=eeg.device), sel].unsqueeze(1)
            # [B, 1, ch, 500]

        return eeg.unsqueeze(2)                # [B, N_kept, 1, ch, 500]

    def forward(self, sample):
        video = sample["video"]
        eeg = sample["eeg"]

        # EEG branch — windowed (train: 1 window per sample; eval: all windows
        # forwarded and logit-averaged for trial-level prediction).
        eeg_win = self._prepare_eeg(eeg)                # [B, N_seg, 1, ch, 500]
        B, N_seg = eeg_win.shape[:2]
        eeg_flat = eeg_win.reshape(B * N_seg, 1, eeg_win.size(-2), eeg_win.size(-1))
        eeg_logit_flat = self.eeg_net(eeg_flat)         # [B*N_seg, n_classes]
        eeg_logit = eeg_logit_flat.view(B, N_seg, -1).mean(dim=1)   # [B, n_classes]

        # Vision branch — per-frame ViT, average logits across T frames
        # (matches EAV's published protocol — they extract per-frame
        # predictions and average).
        frames_flat, B, T = self._prepare_video_frames(video)
        per_frame_logit = self._vit_forward(frames_flat)        # [B*T, n_classes]
        vid_logit = per_frame_logit.view(B, T, -1).mean(dim=1)  # [B, n_classes]

        # Late fusion
        fused = torch.cat([eeg_logit, vid_logit], dim=-1)
        logits = self.fuse(fused)                       # [B, n_classes]

        # V/A reshape for emognition/mdmer
        if self.args.dataset in ("emognition", "mdmer") and self.num_classes > self.num_value:
            x_v = logits[:, :self.num_value].unsqueeze(-1)
            x_a = logits[:, self.num_value:].unsqueeze(-1)
            logits = torch.cat((x_v, x_a), dim=-1)

        return logits
