"""
Milmer (Wang et al., arXiv 2502.00547, 2025) baseline reimplementation for
VEMT comparison.

Verified Milmer protocol (cross-checked against `code/model.py` +
`code/EEG_preprocessing.ipynb` + `config/multi_instance.json`):

EEG (`code/EEG_preprocessing.ipynb`, `code/model.py`)
  - Source: DEAP, 128 Hz, 32 channels, bandpass 1-50 Hz + ICA.
  - Each trial → 20 non-overlapping 3-sec segments of (32 channels, 384
    time samples). The model input shape is (B, channels=32, time=384).
  - Encoder: `LayerNorm(384) → Linear(384, 768) → ReLU`. Each channel's
    384 time samples become a 768-dim token; transformer treats CHANNELS
    as the sequence dimension.

Face encoder (`config/multi_instance.json`, `code/model.py`)
  - `MahmoudWSegni/swin-tiny-patch4-window7-224-finetuned-face-emotion-v12`
    (public on HF, 28-class face emotion finetune).
  - For each of K=10 face instances, Swin returns `last_hidden_state`
    [B, 49, 768] — patch tokens are kept, NOT just pooler_output.
  - Stacked: [B, 10, 49, 768].

MIL aggregation (`select_instances` with `attention_weighted_topk`)
  - Flatten each instance's patch tokens: [B, 10, 49, 768] → [B, 10, 49*768=37632].
  - `topk_value_proj = nn.Linear(37632, 37632)` followed by `tanh` —
    ⚠ this one layer is ~1.4B parameters (≈5.6 GB weight tensor in fp32);
    fits per-GPU in a 4×49 GB DDP configuration with batch_size 3 per
    process (≈22.6 GB for weight + grad + Adam moments).
  - `topk_weight_proj = nn.Linear(37632, 1)` → softmax → topk(3) indices.
  - Top-3 instances kept WITH their full patch tokens → [B, 3, 49, 768] →
    reshape to [B, 147, 768] (= 3 × 49 patches × 768).

Cross-attention "fusion" (`fusion_type='cross_attention'`)
  - 147 LEARNABLE QUERY TOKENS (not video/EEG — Perceiver-style).
  - Cross-attend queries → 147 image patch tokens → refined 147 image
    embeddings (no V↔E cross-attention here; both modalities mix only
    in the joint transformer below).

Joint transformer + CLS
  - Token type embeddings added (0=image, 1=EEG).
  - Concat: [CLS] + [147 image] + [channels EEG] → 2-layer transformer
    (d_model=768, n_heads=12, dim_ff=2048, dropout=0.2).
  - CLS output → dropout(0.1) → Linear(768, n_classes).

Training (`config/multi_instance.json`)
  - AdamW, lr=1e-4, cosine to lr*0.1=1e-5, batch=14, epochs=100, wd=0.01.

Wrapper adaptations to our trial-atomic data pipeline:
  - sample["eeg"] [B, ch, T_seg, P] (CBraMod 200 Hz patched) → faithful
    Milmer EEG pipeline inside the wrapper:
      1. flatten patches → raw EEG at 200 Hz,
      2. bandpass 1-50 Hz (FFT mask; we skip ICA — per-batch ICA is
         infeasible and our dataloader doesn't carry trial-level
         concatenation),
      3. resample 200 Hz → 128 Hz (linear interpolation),
      4. slice into N non-overlapping 384-sample (3-sec) windows.
         For our 10-sec trial → N≈3 windows.
      5. Train: random 1 window per sample. Eval: forward all N windows,
         average logits → trial-level prediction (Milmer's segment-level
         accuracy on its segment-atomic dataset; honest mapping for our
         trial-atomic dataloader).
  - sample["video"] [B, T, C, H, W]: 10 uniformly-spaced frames as MIL
    instances. Our dataset is already face-cropped (MDMER cropped_video),
    so no MediaPipe re-detection. Dense N-clip mode collapsed first.
  - Output reshaped to [B, n_v, 2] for emognition / mdmer V/A.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


_MILMER_HF_FACE = "MahmoudWSegni/swin-tiny-patch4-window7-224-finetuned-face-emotion-v12"


class MilmerModel(nn.Module):
    """Milmer V+EEG fusion. Constructor signature matches VEMT."""

    # MIL bag
    NUM_INSTANCES = 10
    NUM_SELECT = 3
    SWIN_PATCH_TOKENS = 49        # Swin-tiny output [B, 49, 768] for 224x224
    SWIN_DIM = 768

    # Joint transformer dim — Milmer's input_size
    HIDDEN = 768
    N_QUERIES = 147               # 3 instances × 49 patches

    # EEG: bandpass + resample + 3-sec window
    EEG_INPUT_RATE = 200
    EEG_TARGET_RATE = 128
    EEG_TIME_SAMPLES = 384        # 3 sec @ 128 Hz
    EEG_BAND_LO = 1.0
    EEG_BAND_HI = 50.0

    # Joint transformer
    N_HEADS = 12
    DIM_FF = 2048
    N_ENC_LAYERS = 2
    DROPOUT_TF = 0.2
    DROPOUT_CLS = 0.1

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
        self.eeg_channels = eeg_channels

        # --- Face encoder (Swin-Tiny finetuned for face emotion) -----
        self.face_encoder = self._build_face_encoder()

        # --- EEG encoder (Milmer: LayerNorm-over-time → Linear(384,768) → ReLU)
        self.eeg_layernorm = nn.LayerNorm(self.EEG_TIME_SAMPLES)
        self.eeg_proj = nn.Linear(self.EEG_TIME_SAMPLES, self.HIDDEN)

        # --- MIL attention scorer — Milmer's exact `attention_weighted_topk`:
        #     instance_features: [B, 10, 49*768=37632] (flatten patch tokens),
        #     hidden = tanh(Linear(37632, 37632)),
        #     weights = softmax(Linear(37632, 1)),
        #     top-3 instances selected (patch tokens preserved).
        # The first layer is ~1.4B params; fits per-GPU in a 4×49 GB DDP
        # config (~22.6 GB for weight+grad+Adam state, per process).
        _mil_flat = self.SWIN_PATCH_TOKENS * self.SWIN_DIM        # 49 * 768 = 37632
        self.mil_value_proj = nn.Linear(_mil_flat, _mil_flat)
        self.mil_weight_proj = nn.Linear(_mil_flat, 1)

        # --- Cross-attention with learnable queries (Perceiver-style) -
        self.query_tokens = nn.Parameter(torch.zeros(1, self.N_QUERIES, self.HIDDEN))
        nn.init.normal_(self.query_tokens, std=0.02)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.HIDDEN, num_heads=self.N_HEADS,
            dropout=self.DROPOUT_TF, batch_first=True,
        )

        # --- Token type embeddings (0=image, 1=EEG) -----------------
        self.token_type_embeddings = nn.Embedding(2, self.HIDDEN)

        # --- Joint transformer encoder ------------------------------
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.HIDDEN, nhead=self.N_HEADS,
                dim_feedforward=self.DIM_FF,
                dropout=self.DROPOUT_TF, batch_first=True,
            ),
            num_layers=self.N_ENC_LAYERS,
        )

        # --- CLS token + classifier ---------------------------------
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.HIDDEN))
        nn.init.normal_(self.cls_token, std=0.02)
        self.dropout = nn.Dropout(self.DROPOUT_CLS)
        self.classifier = nn.Linear(self.HIDDEN, self.num_classes)

    # ---- Build helpers ---------------------------------------------------
    def _build_face_encoder(self):
        """Try Milmer's face emotion Swin (use_safetensors=True to bypass
        torch 2.1 CVE block on .bin loads); fall back to generic Swin."""
        try:
            from transformers import SwinModel
            for name in (_MILMER_HF_FACE, "microsoft/swin-tiny-patch4-window7-224"):
                try:
                    encoder = SwinModel.from_pretrained(name, use_safetensors=True)
                    tag = "face emotion FT" if name == _MILMER_HF_FACE else "ImageNet base"
                    print(f"[Milmer] face encoder: HF {name} ({tag})")
                    return encoder
                except Exception as e:
                    print(f"[Milmer][WARN] HF load of {name} failed: {e}")
        except ImportError:
            print("[Milmer][WARN] transformers missing — torchvision swin_t.")
        from torchvision.models import swin_t
        return swin_t(weights=None)

    # ---- FFT bandpass (GPU-friendly) ------------------------------------
    @staticmethod
    def _fft_bandpass(x, low_hz, high_hz, fs):
        N = x.size(-1)
        freqs = torch.fft.rfftfreq(N, d=1.0 / fs).to(x.device)
        mask = ((freqs >= low_hz) & (freqs <= high_hz)).to(x.dtype)
        X = torch.fft.rfft(x, dim=-1) * mask
        return torch.fft.irfft(X, n=N, dim=-1)

    # ---- Swin patch-token forward --------------------------------------
    def _swin_patches(self, x):
        """x: [B*K, 3, H, W] → [B*K, 49, 768] (last_hidden_state)."""
        try:
            from transformers.models.swin.modeling_swin import SwinModel
            if isinstance(self.face_encoder, SwinModel):
                return self.face_encoder(pixel_values=x).last_hidden_state
        except ImportError:
            pass
        # torchvision swin_t fallback (no .last_hidden_state — emulate w/
        # final feature map flatten to 49×768).
        feats = self.face_encoder.features(x)
        feats = self.face_encoder.norm(feats)              # [B*K, 7, 7, 768]
        return feats.reshape(feats.size(0), -1, feats.size(-1))

    # ---- Video prep -----------------------------------------------------
    def _prepare_video(self, video):
        """Collapse dense axis, sample K=10 MIL frames. Returns [B*K, C, H, W], B."""
        if video.dim() == 6:                               # dense [B,N,T,C,H,W]
            video = video.mean(dim=1)
        if video.dim() != 5:
            raise ValueError(f"MilmerModel: bad video shape {tuple(video.shape)}")
        B, T, C, H, W = video.shape
        K = self.NUM_INSTANCES
        if T >= K:
            idx = torch.linspace(0, T - 1, steps=K, device=video.device).long()
        else:
            idx = torch.arange(K, device=video.device) % T
        return video[:, idx].reshape(B * K, C, H, W), B

    # ---- EEG prep — Milmer-faithful 4-stage pipeline -------------------
    def _prepare_eeg(self, eeg):
        """[B, ch, T_seg, P] → [B, N_seg, ch, 384] (train: N=1, eval: N=all)."""
        if eeg.dim() == 5:
            eeg = eeg.mean(dim=1)
        if eeg.dim() != 4:
            raise ValueError(f"MilmerModel: bad EEG shape {tuple(eeg.shape)}")
        B, ch, T_seg, P = eeg.shape

        # 1. flatten patches → raw at 200 Hz
        eeg = eeg.reshape(B, ch, T_seg * P)
        # 2. bandpass at 200 Hz (acts as anti-alias for the resample below)
        eeg = self._fft_bandpass(eeg, self.EEG_BAND_LO, self.EEG_BAND_HI,
                                 fs=self.EEG_INPUT_RATE)
        # 3. resample 200 → 128 Hz
        L_in = eeg.size(-1)
        L_out = int(round(L_in * self.EEG_TARGET_RATE / self.EEG_INPUT_RATE))
        if L_out != L_in:
            eeg = F.interpolate(eeg, size=L_out, mode="linear",
                                align_corners=False)
        # 4. slice into N non-overlapping 384-sample (3-sec) windows
        target = self.EEG_TIME_SAMPLES
        if L_out < target:
            eeg = F.interpolate(eeg, size=target, mode="linear",
                                align_corners=False).unsqueeze(1)
            return eeg                                      # [B, 1, ch, 384]
        n_seg = L_out // target
        eeg = eeg[..., :n_seg * target]
        eeg = eeg.reshape(B, ch, n_seg, target).permute(0, 2, 1, 3)  # [B, N, ch, 384]
        # 5. train random window; eval keep all
        if self.training and n_seg > 1:
            sel = torch.randint(0, n_seg, (B,), device=eeg.device)
            eeg = eeg[torch.arange(B, device=eeg.device), sel].unsqueeze(1)
        return eeg

    # ---- MIL attention-weighted top-k selection ------------------------
    def _mil_select(self, instances_patches):
        """Milmer's `attention_weighted_topk` MIL — verbatim.

        instances_patches: [B, 10, 49, 768] → selected [B, 147, 768].
        """
        B, K, P, D = instances_patches.shape
        # 1. Flatten patch tokens into the instance feature (Milmer:
        #    instance_features.view(B, K, -1) → [B, K, 49*768]).
        instance_features = instances_patches.view(B, K, -1)      # [B, K, 49*768]
        # 2. MLP scorer (1.4B params layer):
        hidden = torch.tanh(self.mil_value_proj(instance_features))
        weights = self.mil_weight_proj(hidden).squeeze(-1)        # [B, K]
        weights = F.softmax(weights, dim=1)
        # 3. Top-k indices.
        _, idx = torch.topk(weights, self.NUM_SELECT, dim=1)      # [B, k]
        # 4. Gather selected instances' patch tokens (preserve [P, D]).
        idx_exp = idx.view(B, self.NUM_SELECT, 1, 1).expand(-1, -1, P, D)
        selected = instances_patches.gather(1, idx_exp)           # [B, k, P, D]
        return selected.reshape(B, self.NUM_SELECT * P, D)        # [B, 147, 768]

    # ---- Single-window forward (the trial Milmer trains on) ------------
    def _forward_one_window(self, face_patches, B, eeg_one):
        """face_patches: [B, K=10, 49, 768] (Swin patch tokens for one bag).
        eeg_one: [B, ch, 384]. Returns logits [B, n_classes]."""
        # 1. MIL: select top-3 instances → [B, 147, 768]
        selected = self._mil_select(face_patches)

        # 2. Cross-attention with 147 learnable queries
        Q = self.query_tokens.expand(B, -1, -1)
        image_feats, _ = self.cross_attention(Q, selected, selected)  # [B, 147, 768]

        # 3. EEG: layernorm → linear → ReLU. Per-channel time-sample → 768.
        eeg_feats = self.eeg_layernorm(eeg_one)                 # [B, ch, 384]
        eeg_feats = self.eeg_proj(eeg_feats)                    # [B, ch, 768]
        eeg_feats = F.relu(eeg_feats)

        # 4. Token type embeddings.
        img_type = self.token_type_embeddings(
            torch.zeros(B, 1, dtype=torch.long, device=image_feats.device)
        )                                                       # [B, 1, 768]
        eeg_type = self.token_type_embeddings(
            torch.ones(B, 1, dtype=torch.long, device=image_feats.device)
        )
        image_feats = image_feats + img_type
        eeg_feats = eeg_feats + eeg_type

        # 5. Concat [CLS, image, EEG], joint transformer, CLS classification.
        cls = self.cls_token.expand(B, -1, -1)                  # [B, 1, 768]
        multi = torch.cat([cls, image_feats, eeg_feats], dim=1)
        multi = self.transformer_encoder(multi)
        cls_out = self.dropout(multi[:, 0])                     # [B, 768]
        return self.classifier(cls_out)                         # [B, n_classes]

    # ---- Public forward -------------------------------------------------
    def forward(self, sample):
        # Video: 10 face instances → Swin patches per instance.
        face_flat, B = self._prepare_video(sample["video"])     # [B*K, C, H, W]
        face_patches = self._swin_patches(face_flat)            # [B*K, 49, 768]
        face_patches = face_patches.reshape(
            B, self.NUM_INSTANCES, self.SWIN_PATCH_TOKENS, self.SWIN_DIM
        )

        # EEG → [B, N_seg, ch, 384] (train N=1, eval N=all)
        eeg_win = self._prepare_eeg(sample["eeg"])
        N_seg = eeg_win.size(1)

        # Forward per EEG window; face patches reused across windows.
        logits_per_win = []
        for n in range(N_seg):
            logits_per_win.append(
                self._forward_one_window(face_patches, B, eeg_win[:, n])
            )
        logits = torch.stack(logits_per_win, dim=1).mean(dim=1)  # [B, n_classes]

        # V/A reshape for emognition/mdmer.
        if (self.args.dataset in ("emognition", "mdmer")
                and self.num_classes > self.num_value):
            x_v = logits[:, :self.num_value].unsqueeze(-1)
            x_a = logits[:, self.num_value:].unsqueeze(-1)
            logits = torch.cat((x_v, x_a), dim=-1)
        return logits
