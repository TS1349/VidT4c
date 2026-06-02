import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .CBraMod import CBraMod_Model


# ---- Video backbone registry ---------------------------------------------------
# Maps --vemt_video string -> import path of the wrapper class that implements
# the AdaMAE/VideoMAE VisionTransformer interface (forward(x, return_feat,
# return_tokens) + .blocks / .norm / .head + forward_features_until/from for
# the dense cache split). Adding a new backbone = adding one row here and
# implementing the wrapper in models/vemt/modeling_<name>.py.
_VIDEO_BACKBONE_REGISTRY = {
    "ViViT":    ("modeling_vivit",      "VivitVisionTransformer"),
    "TSF":      ("modeling_tsf",        "TSFVisionTransformer"),
    "Swin":     ("modeling_swin",       "SwinVisionTransformer"),
    "AdaMAE":   ("modeling_finetune_v0", "VisionTransformer"),
    "VideoMAE": ("modeling_finetune",   "VisionTransformer"),
}


def _build_video_backbone(args, image_size, output_dim, embed_dim):
    """Instantiate the video backbone selected by args.vemt_video.

    Keeps vemt.py's __init__ readable: the per-backbone import + class lookup
    lives here, not inline. All registered wrappers share the same kwargs
    contract (see _VIDEO_BACKBONE_REGISTRY docstring).
    """
    import importlib
    name = getattr(args, "vemt_video", "VideoMAE")
    if name not in _VIDEO_BACKBONE_REGISTRY:
        name = "VideoMAE"  # legacy default
    module_name, class_name = _VIDEO_BACKBONE_REGISTRY[name]
    module = importlib.import_module(f".{module_name}", package=__package__)
    cls = getattr(module, class_name)
    return cls(
        args=args,
        img_size=image_size,
        patch_size=16,
        in_chans=3,
        output_dim=output_dim,
        all_frames=32,
        embed_dim=embed_dim,
        depth=12,
        num_heads=12,
        tubelet_size=2,
    )

class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCNLayer, self).__init__()

        self.layers = nn.Sequential(nn.Linear(input_dim, output_dim))
    
    def forward(self, x, adj):
        x = torch.matmul(adj, x)
        x = self.layers(x)
        return x

class GatedGCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GatedGCNLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.edge_gate = nn.Linear(input_dim, input_dim)

    def forward(self, x, adj):
        # x: [B, N, D]

        gate = torch.sigmoid(self.edge_gate(x))  # [B, N, D]

        x = torch.matmul(adj, x * gate)  # gating message passing
        x = self.linear(x)

        return x

class RegionalAttentionPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.scorer = nn.Linear(d_model, 1)
    def forward(self, x, region_indices):
        # x: [B, C, D]
        regions = []
        for ch_idxs in region_indices:
            xr = x[:, ch_idxs, :]                # [B, k, D]
            score = self.scorer(xr)              # [B, k, 1]
            attn = torch.softmax(score, dim=1)
            rfeat = (attn * xr).sum(dim=1)       # [B, D]
            regions.append(rfeat)
        return torch.stack(regions, dim=1)       # [B, R, D]


def build_normalized_adj(x, adj_mask, eps=1e-3, self_loop=1.0, keep_neg=False,
                         binary_adj=False, temporal_weight=None):
    """
    Compute a symmetric normalized adjacency matrix from node features and a mask.

    Args:
        x: Tensor, shape [B, N, D] - Node features
        adj_mask: Tensor, shape [N, N] - Binary mask for allowed connections
        eps: float - Minimum degree clamp to avoid division by zero
        self_loop: float - Value to add to self-connections (I)
        keep_neg: bool - If False, negative similarities are clamped to zero
        binary_adj: If True, use the structural mask directly as adjacency
            (skip cosine similarity weighting). Useful when cross-modal
            cosine sim is unreliable due to modality gap — preserves the
            structural prior encoded in adj_mask (e.g. time-paired video↔EEG).
        temporal_weight: Optional [N, N] tensor that multiplies the similarity
            matrix BEFORE masking/normalization. Used to apply a Gaussian
            temporal-distance prior (clip-clip edges between far-apart clips
            are downweighted). Non-clip rows/cols should be 1.0 to leave them
            unaffected.

    Returns:
        A: Tensor, shape [B, N, N] - Symmetric normalized adjacency
    """
    if binary_adj:
        # Skip cosine sim. Mask itself is the (pre-normalization) adjacency.
        # Broadcast [N, N] -> [B, N, N] for downstream batch ops.
        S = adj_mask.float().to(x.device).unsqueeze(0).expand(x.size(0), -1, -1).contiguous()
    else:
        # L2-normalize
        x_norm = F.normalize(x, p=2, dim=-1)  # [B, N, D]
        # Similarity matrix
        S = torch.matmul(x_norm, x_norm.transpose(1, 2))  # [B, N, N]

        if not keep_neg:
            S = S.clamp_min(0.0)

        S = torch.where(adj_mask.bool().unsqueeze(0), S, torch.zeros_like(S))

    # Temporal-distance modulation (broadcast over batch). Multiplies the
    # similarity matrix so clip-clip edges between distant clips are softened
    # by the Gaussian factor while region/global edges (weight=1) stay intact.
    if temporal_weight is not None:
        S = S * temporal_weight.to(S.device).unsqueeze(0)

    # Symmetrize
    S = 0.5 * (S + S.transpose(1, 2))

    # Add self-loop connections
    if self_loop and self_loop > 0:
        I = torch.eye(S.size(1), device=S.device).unsqueeze(0)
        S = S + self_loop * I

    # Symmetric normalization: A = D^{-1/2} S D^{-1/2}
    deg = S.sum(dim=-1).clamp_min(eps)
    D_inv_sqrt = deg.pow(-0.5)
    D_inv_sqrt = torch.diag_embed(D_inv_sqrt)
    A = D_inv_sqrt @ S @ D_inv_sqrt

    # Remove NaNs or Infs for safety
    A = torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

    return A


class GCN(nn.Module):
    def __init__(self, args, input_dim, output_dim, type="local"):
        super().__init__()
        self.type = type
        self.args = args

        self.gcn1 = GCNLayer(input_dim, input_dim)
        self.gcn2 = GCNLayer(input_dim, output_dim)
        # Non-linearity + norm between gcn1 and gcn2 (the chain was previously
        # linear-only → effectively 1-layer expressivity). GeLU + LayerNorm
        # bring back full 2-layer GCN expressivity.
        self.gcn1_norm = nn.LayerNorm(input_dim)

        if self.type == "region":
            self.weight = nn.Parameter(torch.full((output_dim,), -1.0))

            # Learnable Gaussian std for the temporal-distance adjacency prior
            # (used only when --gcn_temporal_adj is on). Stored in log space for
            # positivity. Initial sigma=3 (in clip-index units) downweights edges
            # between clips ~6 apart by exp(-2) ≈ 0.14 — gentle locality bias.
            self.log_sigma = nn.Parameter(torch.log(torch.tensor(3.0)))

    def compute_local_adj(self, B, C, region_indices, device):
        """Build adjacency mask for local EEG channels."""
        adj_mask = torch.zeros(C, C, device=device)
        if region_indices is None:
            # fully-connected, except self-loop
            adj_mask = torch.ones(C, C, device=device) - torch.eye(C, device=device)
        else:
            for ch_group in region_indices:
                for i in ch_group:
                    for j in ch_group:
                        if i != j:
                            adj_mask[i, j] = 1.0
        return adj_mask

    def compute_region_adj(self, N, device, proto_dim_sizes=(), num_video_nodes=1, num_eeg_nodes=1):
        """Modality-aware adjacency for video clips + EEG clips + brain regions.

        Node layout (in order):
          [0 : K_v)                 = video clip nodes
          [K_v : K_v + K_e)         = EEG clip nodes
          [K_v + K_e : n_main)      = brain region nodes
          [n_main : N)              = optional class prototypes (legacy)

        Edges:
          - video clips <-> video clips: fully connected (no self-loop), when K_v > 1
          - EEG clips   <-> EEG clips:   fully connected (no self-loop), when K_e > 1
          - cross-modal: when K_v == K_e, paired by index (video[i] <-> EEG[i],
            same time window). Otherwise (asymmetric K_v != K_e, e.g. pooled
            single EEG node), all video clips <-> all EEG nodes.
          - all clip globals <-> brain regions
          - regions <-> regions (no self-loop)
        """
        K_v = max(1, int(num_video_nodes))
        K_e = max(1, int(num_eeg_nodes))
        num_protos = sum(proto_dim_sizes)
        n_main = N - num_protos

        eeg_start = K_v
        eeg_end = K_v + K_e
        region_start = eeg_end

        adj_mask = torch.zeros(N, N, device=device)

        # video clip <-> video clip
        if K_v > 1:
            adj_mask[:K_v, :K_v] = 1
            diag = torch.arange(K_v, device=device)
            adj_mask[diag, diag] = 0

        # EEG clip <-> EEG clip
        if K_e > 1:
            adj_mask[eeg_start:eeg_end, eeg_start:eeg_end] = 1
            diag_e = torch.arange(eeg_start, eeg_end, device=device)
            adj_mask[diag_e, diag_e] = 0

        # video <-> EEG cross-modal
        if K_v == K_e:
            # Symmetric per-clip: pair by index (video[i] <-> EEG[i])
            v_idx = torch.arange(K_v, device=device)
            e_idx = v_idx + eeg_start
            adj_mask[v_idx, e_idx] = 1
            adj_mask[e_idx, v_idx] = 1
        else:
            # Asymmetric (e.g. one side pooled): all video <-> all EEG
            adj_mask[:K_v, eeg_start:eeg_end] = 1
            adj_mask[eeg_start:eeg_end, :K_v] = 1

        # all clip globals <-> regions; region <-> region
        if n_main > region_start:
            adj_mask[:region_start, region_start:n_main] = 1
            adj_mask[region_start:n_main, :region_start] = 1
            adj_mask[region_start:n_main, region_start:n_main] = 1
            ridx = torch.arange(region_start, n_main, device=device)
            adj_mask[ridx, ridx] = 0

        # legacy prototype path (always inactive after cleanup; signature kept)
        if num_protos > 0:
            ps = n_main
            adj_mask[:region_start, ps:N] = 1
            adj_mask[ps:N, :region_start] = 1
            offset = ps
            for k in proto_dim_sizes:
                if k > 1:
                    idx = torch.arange(offset, offset + k - 1, device=device)
                    adj_mask[idx, idx + 1] = 1
                    adj_mask[idx + 1, idx] = 1
                offset += k

        return adj_mask

    def _compute_temporal_weight(self, node_time_pos, device):
        """Gaussian temporal-distance weight matrix for clip-clip edges.

        node_time_pos: [N] long tensor; -1 marks non-temporal nodes (regions).
        Returns: [N, N] float weight, 1.0 for edges involving any non-temporal
        node, exp(-|t_i - t_j|² / 2σ²) for pure clip-clip edges.
        """
        t = node_time_pos.to(device).float()
        has_time = (t >= 0)
        both_time = has_time.unsqueeze(0) & has_time.unsqueeze(1)
        t_diff_sq = (t.unsqueeze(0) - t.unsqueeze(1)) ** 2
        sigma = self.log_sigma.exp().clamp(min=0.5)
        gauss = torch.exp(-t_diff_sq / (2.0 * sigma * sigma))
        # Non-temporal pairs (region-anything): weight 1.0 → no modulation.
        return torch.where(both_time, gauss, torch.ones_like(gauss))

    def forward(self, x, region_indices=None, proto_dim_sizes=(),
                num_video_nodes=1, num_eeg_nodes=1, node_time_pos=None):
        if self.type == "local":
            B, C, D = x.shape
            adj_mask = self.compute_local_adj(B, C, region_indices, x.device)
            adj = build_normalized_adj(x, adj_mask, eps=1e-3, self_loop=0.0, keep_neg=True)
            x1 = self.gcn1(x, adj)
            x1 = F.gelu(x1)
            x1 = self.gcn1_norm(x1)
            x1 = 0.5 * x1 + 0.5 * x  # residual connection
            x1 = self.gcn2(x1, adj)
            return x1  # [B, C, D_out]

        elif self.type == "region":
            B, N, D = x.shape
            K_v = max(1, int(num_video_nodes))
            K_e = max(1, int(num_eeg_nodes))

            adj_mask = self.compute_region_adj(
                N, x.device,
                proto_dim_sizes=proto_dim_sizes,
                num_video_nodes=K_v,
                num_eeg_nodes=K_e,
            )

            # Optional temporal-distance adjacency: multiplies content sim by
            # a Gaussian of |t_i - t_j| for clip-clip edges (regions unaffected).
            temporal_weight = None
            if getattr(self.args, 'gcn_temporal_adj', False) and node_time_pos is not None:
                temporal_weight = self._compute_temporal_weight(node_time_pos, x.device)

            _binary = getattr(self.args, 'gcn_binary_adj', False)
            adj = build_normalized_adj(
                x, adj_mask,
                eps=1e-3, self_loop=0.0, keep_neg=True,
                binary_adj=_binary,
                temporal_weight=temporal_weight,
            )

            x1 = self.gcn1(x, adj)
            x1 = F.gelu(x1)
            x1 = self.gcn1_norm(x1)
            x1 = 0.5 * x1 + 0.5 * x
            x1 = self.gcn2(x1, adj)

            # Pool post-GCN node features per modality.
            # Video clips at [0:K_v) → mean → video global.
            # EEG clips at [K_v:K_v+K_e) → mean → EEG global.
            if K_v > 1:
                global_f1 = x1[:, :K_v, :].mean(dim=1)
            else:
                global_f1 = x1[:, 0, :]
            if K_e > 1:
                global_f2 = x1[:, K_v:K_v + K_e, :].mean(dim=1)
            else:
                global_f2 = x1[:, K_v, :]
            # stacked = torch.stack([global_f1, global_f2], dim=1)  # [B, 2, D_out]

            # out = torch.bmm(self.weight.expand(B, -1, -1), stacked).squeeze(1)

            # w = torch.softmax(self.weight, dim=-1) # Restrict as 1
            # out = torch.bmm(w.expand(B, -1, -1), stacked).squeeze(1)

            w = torch.sigmoid(self.weight)
            out = (1 - w) * global_f2 + w * global_f1

            return out  # [B, D_out]

        else:
            raise ValueError(f"Invalid GCN type: {self.type}")


class RegionalAttentionPool(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.scorer = nn.Linear(d_model, 1)
    def forward(self, x, region_indices):
        regions = []
        for ch_idxs in region_indices:
            xr = x[:, ch_idxs, :]  # [B, k, D]
            score = self.scorer(xr)
            attn = torch.softmax(score, dim=1)
            rfeat = (attn * xr).sum(dim=1)
            regions.append(rfeat)
        return torch.stack(regions, dim=1) # [B, R, D]


class FiLM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Linear(dim, dim)
        self.beta = nn.Linear(dim, dim)

    def forward(self, x, cond):
        # x: video_f [B, D]
        # cond: eeg_f [B, D]

        gamma = self.gamma(cond)
        beta = self.beta(cond)

        return gamma * x + beta

class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)

        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query, key_value):
        q = self.q_proj(query).unsqueeze(1)
        k = self.k_proj(key_value).unsqueeze(1)
        v = self.v_proj(key_value).unsqueeze(1)

        out, _ = self.attn(q, k, v)
        out = self.norm(out + q)

        return out.squeeze(1)

# Version 2 with residual gating
class NodeGate_residual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Linear(dim, dim)

    def forward(self, x):
        g = torch.sigmoid(self.gate(x))  # [B, N, D]
        return x * (1 + g)

# Version 1 with only scalar
class NodeGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Linear(dim, 1)

    def forward(self, x):
        # x: [B, N, D]
        g = torch.sigmoid(self.gate(x))  # [B, N, 1]
        out_gating = x * g
        return out_gating

class VEMT(nn.Module):
    def __init__(self, args,
                 output_dim,
                 image_size,
                 eeg_channels,
                 frequency_bins,
                 ):

        super().__init__()
        self.args = args

        self.image_size = image_size
        self.output_dim = output_dim
        self.spectrogram_size = (128, 256)
        self.embed_dim = 768
        self.hs = self.embed_dim * 2
        self.num_classes = output_dim[0] * output_dim[1]
        self.num_class = output_dim[0]
        self.in_channel = eeg_channels

        self.frequency_bins = frequency_bins

        if self.args.fusion == 'FiLM':
            self.node_gate = NodeGate(dim=self.embed_dim)

        if output_dim[1] == 1:
            self.output_dim = (output_dim[0],)

        if self.args.set_video_only or self.args.eeg_signal:
            # Build the video backbone selected by --vemt_video.
            # ViViT/TSF/Swin self-load pretrained weights inside their wrapper;
            # AdaMAE/VideoMAE go through load_flexible in runner.py.
            self.video_model = _build_video_backbone(
                args=args,
                image_size=image_size,
                output_dim=output_dim,
                embed_dim=self.embed_dim,
            )


        if self.args.set_eeg_only or self.args.eeg_signal:
            # EEG backbone choice (--eeg_backbone): default 'cbramod' keeps the
            # legacy path unchanged. 'reve' loads HF brain-bzh/reve-base via a
            # wrapper that mirrors CBraMod_Model's constructor + forward
            # contract, so downstream GCN / fusion / classifier code is
            # untouched. REVE requires HF gating acceptance + login (see
            # models/vemt/REVE.py docstring for setup).
            _eeg_backbone = getattr(args, 'eeg_backbone', 'cbramod')
            if _eeg_backbone == 'reve':
                from .REVE import REVE_Model
                self.eeg_model = REVE_Model(args, output_dim=output_dim, in_chans=self.in_channel)
            else:
                self.eeg_model = CBraMod_Model(args, output_dim=output_dim, in_chans=self.in_channel)
            # self.eeg_model = AudioMamba(args=args, spectrogram_size=self.spectrogram_size, depth=24, channels=self.in_channel, output_dim=output_dim)
            # self.eeg_model = Crossformer_Model(args, output_dim=output_dim, enc_in=self.in_channel, seq_len=256) # CrossFormer
            # self.eeg_model = PatchTST_Model(args, output_dim=output_dim, enc_in=self.in_channel, seq_len=256) # PatchTST

        # E+V Murged classifier Head
        if self.args.eeg_signal:
            if self.args.gcn:
                # self.eeg_feat_proj = nn.Linear(200, self.embed_dim)
                # Per-channel EEG feature source for GCN region nodes:
                #   'stft' (default, legacy): hand-crafted STFT [B, ch, F*T] → Linear(F*T, 768)
                #     Random-init 25M projection. No CBraMod pretrain benefit. Channel-independent.
                #   'cbramod': CBraMod patch_embedding's pre-positional output
                #     [B, ch, T_seg=10, 200] (conv on raw EEG + FFT projection, BEFORE the
                #     (19,7) positional encoding conv that mixes channels). Pretrained, channel-
                #     independent.
                #
                # Projection kind (only when --gcn_region_eeg_source cbramod):
                #   'linear' (default): mean-pool T_seg → Linear(200, 768). 154K params. Simple.
                #   'conv': preserve T_seg, Conv1d(200→768, k=3) → GELU → Conv1d(768→768, k=3)
                #     → AdaptiveAvgPool1d. Learns temporal patterns per channel. ~2.3M params.
                _eeg_src = getattr(args, 'gcn_region_eeg_source', 'stft')
                _eeg_proj_kind = getattr(args, 'gcn_region_eeg_proj', 'linear')
                # Backbone-native per-channel feature dim coming out of the
                # EEG backbone's return_per_channel_pre path.
                # CBraMod: 200 (its d_model). REVE-base: 512 (config.embed_dim).
                _eeg_backbone_dim = (
                    512 if getattr(args, 'eeg_backbone', 'cbramod') == 'reve' else 200
                )
                if _eeg_src == 'cbramod':
                    if _eeg_proj_kind == 'conv':
                        _ks = 3
                        self.eeg_feat_proj = nn.Sequential(
                            nn.Conv1d(_eeg_backbone_dim, self.embed_dim, _ks, padding=_ks // 2),
                            nn.GELU(),
                            nn.Conv1d(self.embed_dim, self.embed_dim, _ks, padding=_ks // 2),
                            nn.AdaptiveAvgPool1d(1),
                        )
                    else:
                        self.eeg_feat_proj = nn.Linear(_eeg_backbone_dim, self.embed_dim)
                else:
                    self.eeg_feat_proj = nn.Linear(self.frequency_bins, self.embed_dim)
                self.gcn_local = GCN(args, input_dim=self.embed_dim, output_dim=self.embed_dim, type= "local")
                self.gcn_region = GCN(args, input_dim=self.embed_dim, output_dim=self.num_classes, type = "region")
                self.region_pool = RegionalAttentionPool(d_model=self.embed_dim)

                if self.args.fusion == 'router':
                    if len(self.output_dim) > 1:  # multihead (emognition/mdmer): separate V/A routers
                        _n = self.output_dim[0]
                        _extra = 6 if getattr(args, 'router_entropy', False) else 0  # 3 experts × (H + margin)
                        self.router_mlp_v = nn.Sequential(
                            nn.Linear(_n * 3 + _extra, _n),
                            nn.ReLU(),
                            nn.Dropout(0.1),
                            nn.Linear(_n, 3)
                        )
                        self.router_mlp_a = nn.Sequential(
                            nn.Linear(_n * 3 + _extra, _n),
                            nn.ReLU(),
                            nn.Dropout(0.1),
                            nn.Linear(_n, 3)
                        )
                    else:
                        self.router_mlp = nn.Sequential(
                            nn.Linear(self.num_classes * 3, self.num_classes),
                            nn.ReLU(),
                            nn.Dropout(0.1),
                            nn.Linear(self.num_classes, 3)
                        )

                # Only create auxiliary heads when explicitly needed.
                # Router uses each backbone's native classifier/head instead.
                if getattr(args, 'conf_gate', False):
                    self.eeg_head = nn.Sequential(nn.Linear(768, self.num_classes))
                    self.video_head = nn.Sequential(
                        nn.Linear(768, 200),
                        nn.ELU(),
                        nn.Dropout(0.1),
                        nn.Linear(200, self.num_classes)
                    )

                # Class log-priors for conf_gate logit adjustment.
                # Defaults to zeros (no adjustment); runner.py overrides from training stats.
                if getattr(args, 'conf_gate', False):
                    if self.args.dataset in ('emognition', 'mdmer'):
                        self.register_buffer('log_val_prior', torch.zeros(self.num_class))
                        self.register_buffer('log_aro_prior', torch.zeros(self.num_class))
                    else:
                        self.register_buffer('log_class_prior', torch.zeros(self.num_classes))

                # EEG temporal cross-attention (asymmetric: Q=video global, K/V=EEG segments).
                if getattr(args, 'eeg_temporal_attn', False):
                    self.eeg_temporal_attn_module = nn.MultiheadAttention(
                        embed_dim=self.embed_dim, num_heads=8, batch_first=True,
                        kdim=200, vdim=200  # CBraMod patch_emb d_model=200, Q=video 768
                    )
                    self.eeg_temporal_alpha = nn.Parameter(torch.tensor(-1.0))  # sigmoid(-1) ≈ 0.27

                # Channel-level temporal attention: each EEG channel independently attends
                if getattr(args, 'ch_temporal_attn', False):
                    self.ch_temporal_attn_module = nn.MultiheadAttention(
                        embed_dim=self.embed_dim, num_heads=8, batch_first=True,
                        kdim=200, vdim=200
                    )

                if getattr(args, 'eeg_video_spatial_attn', False):
                    self.eeg_spatial_attn = nn.MultiheadAttention(
                        embed_dim=self.embed_dim, num_heads=8, batch_first=True
                    )
                    self.eeg_spatial_alpha = nn.Parameter(torch.tensor(-4.0))  # sigmoid(-2) ≈ 0.12

                # Cross-modal temporal attention: per-clip video queries attend over
                # EEG temporal patches (CBraMod encoder output, channel-pooled). Each
                # video clip pulls a time-aligned EEG context → residual update.
                # Aligned per-clip features get K-pooled to a single video global so
                # the downstream GCN stays 4-node (avoids per-clip over-smoothing).
                if getattr(args, 'cross_modal_temporal_attn', False):
                    self.cross_modal_temporal_attn_module = nn.MultiheadAttention(
                        embed_dim=self.embed_dim, num_heads=8, batch_first=True,
                        kdim=200, vdim=200,  # CBraMod patch d_model=200
                    )
                    self.cross_modal_temporal_alpha = nn.Parameter(torch.tensor(-1.0))
                    # sigmoid(-1) ≈ 0.27 → conservative residual at init

                self.use_region_importance = False

                if self.args.fusion == 'region':
                    self.use_region_importance = True

                    num_regions = 2 if self.args.dataset == "emognition" else 5
                    self.region_gate_mlp = nn.Sequential(
                        nn.Linear(self.embed_dim * 5, self.embed_dim),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(self.embed_dim, 1)
                    )
            
            else:
                self.classifier = nn.Linear(self.hs, self.num_classes)

            # 2-way router without GCN (EEG vs Video only)
            if self.args.fusion == 'router' and not self.args.gcn:
                self.eeg_head = nn.Sequential(nn.Linear(768, self.num_classes))
                self.video_head = nn.Sequential(
                    nn.Linear(768, 200),
                    nn.ELU(),
                    nn.Dropout(0.1),
                    nn.Linear(200, self.num_classes)
                )
                if len(self.output_dim) > 1:
                    _n = self.output_dim[0]
                    _extra = 4 if getattr(args, 'router_entropy', False) else 0  # 2 experts × (H + margin)
                    self.router_mlp_v = nn.Sequential(
                        nn.Linear(_n * 2 + _extra, _n), nn.ReLU(), nn.Linear(_n, 2)
                    )
                    self.router_mlp_a = nn.Sequential(
                        nn.Linear(_n * 2 + _extra, _n), nn.ReLU(), nn.Linear(_n, 2)
                    )
                    # learnable per-modality temperature for probability-space mixing
                    self.log_T_eeg_v = nn.Parameter(torch.zeros(1))
                    self.log_T_vid_v = nn.Parameter(torch.zeros(1))
                    self.log_T_eeg_a = nn.Parameter(torch.zeros(1))
                    self.log_T_vid_a = nn.Parameter(torch.zeros(1))
                else:
                    self.router_mlp = nn.Sequential(
                        nn.Linear(self.num_classes * 2, self.num_classes),
                        nn.ReLU(), nn.Dropout(0.1), nn.Linear(self.num_classes, 2)
                    )

        self.eeg_proj = nn.Linear(200, 768)

        if self.args.fusion == 'FiLM':
            self.film = FiLM(dim=self.embed_dim)
        elif self.args.fusion == 'attention':
            self.cross_attn = CrossAttentionFusion(self.embed_dim)

        # Clip attention pooling: per-modality scalar score per clip → softmax weighted sum.
        # Two knobs control how non-uniform the softmax weights are:
        #   --clip_attn_init_std  : init std for the Linear(768, 1) scorer (default 0.5)
        #                            larger → broader initial score range
        #   --clip_attn_temp      : divisor on scores before softmax (default 1.0)
        #                            smaller (e.g. 0.1) → sharper softmax peaks
        # Both are knobs because empirically backbone features are smaller than
        # standard ViT estimates (after the frozen path, mean_dev from uniform
        # was ~4e-3 with std=0.1), so we need extra amplification to actually
        # differentiate clips at init and get gradient signal to clip_attn.
        _attn_init_std = float(getattr(args, 'clip_attn_init_std', 0.5))
        _multi_video = (getattr(args, 'num_clips', 1) > 1
                        or getattr(args, 'dense_video_clips', False))
        _multi_eeg = (getattr(args, 'num_clips', 1) > 1)  # EEG has no dense mode
        if getattr(args, 'clip_pool', 'mean') == 'attn':
            if (self.args.set_video_only or self.args.eeg_signal) and _multi_video:
                self.clip_attn_v = nn.Linear(self.embed_dim, 1)
                nn.init.normal_(self.clip_attn_v.weight, std=_attn_init_std)
                nn.init.zeros_(self.clip_attn_v.bias)
            if (self.args.set_eeg_only or self.args.eeg_signal) and _multi_eeg:
                self.clip_attn_e = nn.Linear(self.embed_dim, 1)
                nn.init.normal_(self.clip_attn_e.weight, std=_attn_init_std)
                nn.init.zeros_(self.clip_attn_e.bias)

        # --clip_pool_temporal: temporal-aware refinement of K clip features BEFORE pool.
        # Mirrors the GCN temporal method (PE on nodes + Gaussian-decay on edges),
        # but applied to the pre-pool clip tensor [B, K, D] instead of inside the
        # region GCN. Useful when gcn_video_per_clip=False (single pooled video node):
        # the pooled feature now carries temporally-refined clip context.
        # Adds one learnable scalar (pool_log_sigma); PE is parameter-free.
        if getattr(args, 'clip_pool_temporal', False) and _multi_video:
            self.pool_log_sigma = nn.Parameter(torch.log(torch.tensor(3.0)))

        # Dense video features cache (block-split intermediate). Has two tiers:
        #   * in-memory dict (per-rank): fast hit for repeated access within rank
        #   * on-disk (shared via filesystem across ranks AND across epochs):
        #     primary store. Critical because DistributedSampler reshuffles
        #     rank↔sample assignment each epoch, so an in-memory-only cache
        #     populated by rank A in epoch 0 cannot help rank B in epoch 1.
        # Set self._video_block_cache_dir via setup_dense_cache(dir) to enable
        # the on-disk tier (None = memory only).
        # Keyed by dataset idx; values stored as fp16 CPU [N_actual, P, D].
        self._video_block_cache = {}
        self._video_block_cache_dir = None

    def freeze_backbones(self, video_unfreeze_last_n: int = 0, eeg_unfreeze_last_n: int = 0, eeg_full_unfreeze: bool = False):
        """
        Freeze only pretrained backbone parts.
        Keep classifier / fusion heads trainable.

        Args:
            video_unfreeze_last_n: If >0, leave the last N transformer blocks of
                video_model trainable (along with norm/fc_norm/head). Lets the
                upper layers adapt to emotion while keeping early blocks frozen
                for memory efficiency under K-clip.
            eeg_unfreeze_last_n: Same idea for CBraMod. >0 keeps the last N
                encoder layers (`self.eeg_model.encoder.layers[-N:]`) trainable
                in addition to classifier. 0 = classifier-only (linear probing).
            eeg_full_unfreeze: If True, fully unfreeze CBraMod (encoder +
                patch_embedding + classifier). Overrides eeg_unfreeze_last_n.
                CBraMod is small (~22.5M) so full FT is feasible even under K-clip.
        """

        # 1) Video model freeze
        if hasattr(self, "video_model"):
            for p in self.video_model.parameters():
                p.requires_grad = False

            head = getattr(self.video_model, "head", None)
            if head is not None:
                for p in head.parameters():
                    p.requires_grad = True

            n_unfreeze = max(0, int(video_unfreeze_last_n))
            blocks = getattr(self.video_model, "blocks", None)
            if n_unfreeze > 0 and blocks is not None:
                n_blocks = len(blocks)
                unfreeze_threshold = n_blocks - n_unfreeze
                for i in range(unfreeze_threshold, n_blocks):
                    for p in blocks[i].parameters():
                        p.requires_grad = True
                # post-block norms (AdaMAE: norm/fc_norm; ViViT: norm only)
                for attr in ("norm", "fc_norm"):
                    mod = getattr(self.video_model, attr, None)
                    if mod is not None:
                        for p in mod.parameters():
                            p.requires_grad = True

        # 2) EEG model freeze
        # Default: classifier only trainable.
        # eeg_full_unfreeze: everything trainable (encoder + patch_embedding + classifier).
        # else eeg_unfreeze_last_n > 0: also unfreeze last N CBraMod encoder layers.
        if hasattr(self, "eeg_model"):
            if eeg_full_unfreeze:
                for p in self.eeg_model.parameters():
                    p.requires_grad = True
            else:
                for name, p in self.eeg_model.named_parameters():
                    if name.startswith("classifier"):
                        p.requires_grad = True
                    else:
                        p.requires_grad = False

                n_unfreeze_e = max(0, int(eeg_unfreeze_last_n))
                encoder = getattr(self.eeg_model, "encoder", None)
                if n_unfreeze_e > 0 and encoder is not None:
                    layers = getattr(encoder, "layers", None)
                    if layers is not None:
                        n_layers = len(layers)
                        unfreeze_threshold_e = n_layers - n_unfreeze_e
                        for i in range(unfreeze_threshold_e, n_layers):
                            for p in layers[i].parameters():
                                p.requires_grad = True

        # 3) VEMT fusion / GCN heads keep trainable
        head_module_names = [
            "eeg_feat_proj",
            "gcn_local",
            "gcn_region",
            "region_pool",
            "region_gate_mlp",
            "eeg_head",
            "video_head",
            "eeg_temporal_attn_module",
            "ch_temporal_attn_module",
            "eeg_spatial_attn",
            "cross_modal_temporal_attn_module",
            "classifier",  # VEMT classifier when gcn=False
            "film",
            "cross_attn",
        ]

        for module_name in head_module_names:
            if hasattr(self, module_name):
                module = getattr(self, module_name)
                for p in module.parameters():
                    p.requires_grad = True

    def init_weights_classify(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def apply_region_importance(self, eeg_region, video_f, eeg_f):
        """
        eeg_region: [B, R, D]
        video_f:    [B, D]
        eeg_f:      [B, D]
        """
        context = torch.cat([
            video_f,
            eeg_f,
            video_f * eeg_f,
            torch.abs(video_f - eeg_f)
        ], dim=-1)  # [B, 4D]

        context = context.unsqueeze(1).expand(-1, eeg_region.size(1), -1)  # [B, R, 4D]

        gate_in = torch.cat([eeg_region, context], dim=-1)  # [B, R, 5D]
        alpha = torch.sigmoid(self.region_gate_mlp(gate_in)).squeeze(-1)  # [B, R]

        eeg_region = eeg_region * (1.0 + alpha.unsqueeze(-1))

        return eeg_region, alpha

    def _mix_logits(self, logit_eeg, logit_video, logit_gcn):
        """
        logit_eeg, logit_video, logit_gcn: [B, C]
        """
        router_in = torch.cat([logit_eeg, logit_video, logit_gcn], dim=-1)# [B, 3C]
        router_logits = self.router_mlp(router_in)
        mix_w = torch.softmax(router_logits, dim=-1)  # [B, 3]

        w_eeg = mix_w[:, 0].unsqueeze(-1)  # [B, 1]
        w_video = mix_w[:, 1].unsqueeze(-1)  # [B, 1]
        w_gcn = mix_w[:, 2].unsqueeze(-1)  # [B, 1]

        final_logit = (
                w_eeg * logit_eeg +
                w_video * logit_video +
                w_gcn * logit_gcn
        )

        return final_logit, mix_w, router_logits

    @staticmethod
    def _router_extra_features(probs):
        """Per-expert entropy + margin features for router input.

        Args:
            probs: list of [B, C] probability tensors (already softmax-ed)
        Returns:
            [B, 2 * len(probs)] — interleaved [H_0, margin_0, H_1, margin_1, ...]
        """
        feats = []
        for p in probs:
            H = -(p * torch.log(p.clamp_min(1e-9))).sum(-1, keepdim=True)  # [B, 1]
            top2 = p.topk(min(2, p.size(-1)), dim=-1).values               # [B, 2]
            margin = (top2[:, 0] - top2[:, 1]).unsqueeze(-1)               # [B, 1]
            feats.extend([H, margin])
        return torch.cat(feats, dim=-1)  # [B, 2 * n_experts]

    def _mix_logits_va(self, logit_eeg, logit_video, logit_gcn):
        """V/A separate 3-way routing. logits are [B, n_v+n_a] flat.

        When stop_gradient=True: probability-space mixing → returns [B, C, 2] log-probs.
        Otherwise: logit-space mixing → returns [B, n_v+n_a] logits.
        """
        n_v = self.output_dim[0]
        eeg_v, eeg_a   = logit_eeg[:, :, 0],   logit_eeg[:, :, 1]
        vid_v, vid_a   = logit_video[:, :, 0],  logit_video[:, :, 1]
        gcn_v, gcn_a   = logit_gcn[:, :n_v],   logit_gcn[:, n_v:]

        use_prob    = getattr(self.args, 'stop_gradient', False)
        use_entropy = getattr(self.args, 'router_entropy', False)

        p_eeg_v = F.softmax(eeg_v, dim=-1)
        p_vid_v = F.softmax(vid_v, dim=-1)
        p_gcn_v = F.softmax(gcn_v, dim=-1)
        p_eeg_a = F.softmax(eeg_a, dim=-1)
        p_vid_a = F.softmax(vid_a, dim=-1)
        p_gcn_a = F.softmax(gcn_a, dim=-1)

        if use_prob:
            router_in_v = torch.cat([p_eeg_v, p_vid_v, p_gcn_v], dim=-1)
            router_in_a = torch.cat([p_eeg_a, p_vid_a, p_gcn_a], dim=-1)
        else:
            router_in_v = torch.cat([eeg_v, vid_v, gcn_v], dim=-1)
            router_in_a = torch.cat([eeg_a, vid_a, gcn_a], dim=-1)

        if use_entropy:
            extra_v = self._router_extra_features([p_eeg_v, p_vid_v, p_gcn_v])
            extra_a = self._router_extra_features([p_eeg_a, p_vid_a, p_gcn_a])
            router_in_v = torch.cat([router_in_v, extra_v], dim=-1)
            router_in_a = torch.cat([router_in_a, extra_a], dim=-1)

        rlogits_v = self.router_mlp_v(router_in_v)  # [B, 3]
        rlogits_a = self.router_mlp_a(router_in_a)  # [B, 3]
        w_v = torch.softmax(rlogits_v, dim=-1)
        w_a = torch.softmax(rlogits_a, dim=-1)

        if use_prob:
            p_mix_v = w_v[:, 0:1]*p_eeg_v + w_v[:, 1:2]*p_vid_v + w_v[:, 2:3]*p_gcn_v
            p_mix_a = w_a[:, 0:1]*p_eeg_a + w_a[:, 1:2]*p_vid_a + w_a[:, 2:3]*p_gcn_a
            log_p = torch.stack([
                torch.log(p_mix_v.clamp_min(1e-9)),
                torch.log(p_mix_a.clamp_min(1e-9))
            ], dim=-1)  # [B, C, 2]
            mix_w        = torch.cat([w_v, w_a], dim=-1)               # [B, 6]
            router_logits = torch.cat([rlogits_v, rlogits_a], dim=-1)  # [B, 6]
            return log_p, mix_w, router_logits
        else:
            final_v = w_v[:, 0:1]*eeg_v + w_v[:, 1:2]*vid_v + w_v[:, 2:3]*gcn_v
            final_a = w_a[:, 0:1]*eeg_a + w_a[:, 1:2]*vid_a + w_a[:, 2:3]*gcn_a
            final        = torch.cat([final_v, final_a], dim=-1)      # [B, n_v+n_a]
            mix_w        = torch.cat([w_v, w_a], dim=-1)               # [B, 6]
            router_logits = torch.cat([rlogits_v, rlogits_a], dim=-1)  # [B, 6]
            return final, mix_w, router_logits

    def _mix_logits_va_2way(self, logit_eeg, logit_video):
        """
        V/A separate 2-way routing with probability-space mixing.

        Input:
            logit_eeg:   [B, C, 2]  where [:, :, 0] = Valence, [:, :, 1] = Arousal
            logit_video: [B, C, 2]

        Output:
            log_p:         [B, C, 2]  log probability for NLLLoss
            mix_w:         [B, 4]     [w_eeg_v, w_video_v, w_eeg_a, w_video_a]
            router_logits: [B, 4]     [r_eeg_v, r_video_v, r_eeg_a, r_video_a]
        """

        eeg_v = logit_eeg[:, :, 0]
        eeg_a = logit_eeg[:, :, 1]
        vid_v = logit_video[:, :, 0]
        vid_a = logit_video[:, :, 1]

        # Expert probabilities
        p_eeg_v = F.softmax(eeg_v, dim=-1)
        p_eeg_a = F.softmax(eeg_a, dim=-1)
        p_vid_v = F.softmax(vid_v, dim=-1)
        p_vid_a = F.softmax(vid_a, dim=-1)

        # Router predicts expert weights from expert probabilities (+ optional entropy/margin)
        router_in_v = torch.cat([p_eeg_v, p_vid_v], dim=-1)
        router_in_a = torch.cat([p_eeg_a, p_vid_a], dim=-1)
        if getattr(self.args, 'router_entropy', False):
            extra_v = self._router_extra_features([p_eeg_v, p_vid_v])
            extra_a = self._router_extra_features([p_eeg_a, p_vid_a])
            router_in_v = torch.cat([router_in_v, extra_v], dim=-1)
            router_in_a = torch.cat([router_in_a, extra_a], dim=-1)
        rlogits_v = self.router_mlp_v(router_in_v)
        rlogits_a = self.router_mlp_a(router_in_a)

        w_v = F.softmax(rlogits_v, dim=-1)
        w_a = F.softmax(rlogits_a, dim=-1)

        # Probability mixture
        p_mix_v = w_v[:, 0:1] * p_eeg_v + w_v[:, 1:2] * p_vid_v
        p_mix_a = w_a[:, 0:1] * p_eeg_a + w_a[:, 1:2] * p_vid_a

        log_p_v = torch.log(p_mix_v.clamp_min(1e-9))
        log_p_a = torch.log(p_mix_a.clamp_min(1e-9))

        log_p = torch.stack([log_p_v, log_p_a], dim=-1)
        mix_w = torch.cat([w_v, w_a], dim=-1)
        router_logits = torch.cat([rlogits_v, rlogits_a], dim=-1)

        return log_p, mix_w, router_logits

    def _mix_logits_2way(self, logit_eeg, logit_video):
        """Single-head 2-way routing (no GCN)."""
        router_in = torch.cat([logit_eeg, logit_video], dim=-1)
        router_logits = self.router_mlp(router_in)
        mix_w = torch.softmax(router_logits, dim=-1)
        final = mix_w[:, 0:1]*logit_eeg + mix_w[:, 1:2]*logit_video
        return final, mix_w, router_logits

    def _gated_entropy(self, logit, T=1.0):
        """Logit-adjusted (class-prior debiased), temperature-scaled entropy. Multihead-aware."""
        adj = logit / T
        if adj.size(-1) == 2 * self.num_class and hasattr(self, 'log_val_prior'):
            adj_v = adj[:, :self.num_class] - self.log_val_prior.unsqueeze(0)
            adj_a = adj[:, self.num_class:] - self.log_aro_prior.unsqueeze(0)
            H_v = -(F.softmax(adj_v, -1) * F.log_softmax(adj_v, -1)).sum(-1)
            H_a = -(F.softmax(adj_a, -1) * F.log_softmax(adj_a, -1)).sum(-1)
            return 0.5 * (H_v + H_a)
        if hasattr(self, 'log_class_prior'):
            adj = adj - self.log_class_prior.unsqueeze(0)
        return -(F.softmax(adj, -1) * F.log_softmax(adj, -1)).sum(-1)

    def _va_to_flat(self, logit):
        if logit.dim() == 3 and logit.size(-1) == 2:
            return torch.cat([logit[:, :, 0], logit[:, :, 1]], dim=1)
        return logit

    def _flat_to_va(self, logit):
        if logit.dim() == 2 and self.args.dataset in ('emognition', 'mdmer'):
            n = self.output_dim[0]
            return torch.stack([logit[:, :n], logit[:, n:]], dim=-1)
        return logit

    def _temporal_refine_clips(self, feats):
        """Apply PE + Gaussian temporal-decay self-refinement to K clip features.

        Mirrors the GCN temporal method (--gcn_clip_pe + --gcn_temporal_adj) but
        operates on the K clip tensor BEFORE pool collapses them, so the pooled
        feature carries temporally-aware clip context (usable in pooled-feature
        GCN mode where K clip nodes never reach the region GCN).

        feats: [B, K, D] (3-D only; higher-dim e.g. CBraMod [B,K,Ch,T,D] is skipped).
        Returns: [B, K, D] refined.

        Mechanism:
          1. PE addition  : feats += sin/cos PE(K, D)    — node identity
          2. Cosine-sim adjacency between clips        — content link
          3. Gaussian temporal-decay weight on edges:
             w_ij = exp(-(i-j)^2 / 2σ^2), σ = exp(pool_log_sigma).clamp(0.5)
          4. relu + row-normalize → message passing: refined = adj @ feats

        Only adds 1 learnable scalar (pool_log_sigma); PE is parameter-free.
        """
        if feats.dim() != 3 or feats.size(1) < 2:
            return feats  # nothing to refine
        B, K, D = feats.shape
        device = feats.device

        # 1. PE
        pe = self._sinusoid_pe(K, D, device).unsqueeze(0)  # [1, K, D]
        x = feats + pe

        # 2. Cosine-sim adjacency [B, K, K]
        x_norm = F.normalize(x, dim=-1)
        sim = torch.bmm(x_norm, x_norm.transpose(1, 2))

        # 3. Gaussian temporal-decay
        t = torch.arange(K, device=device, dtype=torch.float32)
        t_diff_sq = (t.unsqueeze(0) - t.unsqueeze(1)) ** 2  # [K, K]
        sigma = self.pool_log_sigma.exp().clamp(min=0.5)
        gauss = torch.exp(-t_diff_sq / (2.0 * sigma * sigma))  # [K, K]

        # 4. Combine + normalize + message pass
        adj = F.relu(sim) * gauss.unsqueeze(0)               # [B, K, K]
        adj = adj / (adj.sum(dim=-1, keepdim=True) + 1e-6)
        refined = torch.bmm(adj, x)                          # [B, K, D]
        return refined

    def _pool_clips(self, feat_flat, logit_flat, B, K, modality='video'):
        """Pool per-clip features and logits across the K dimension.
        modality:   'video' or 'eeg' — selects the attention layer for clip_pool='attn'

        Returns (logit, feat) with K removed, original trailing shape preserved.
        Pool method controlled by self.args.clip_pool ∈ {mean, max, attn}.
        """
        pool = getattr(self.args, 'clip_pool', 'mean')
        feats  = feat_flat.view(B, K, *feat_flat.shape[1:])
        logits = logit_flat.view(B, K, *logit_flat.shape[1:])

        # Optional temporal-aware refinement BEFORE pool (mirrors GCN temporal method).
        # Active only for video modality with K>=2 and 3-D feats (skip CBraMod 5-D).
        if (modality == 'video'
                and getattr(self.args, 'clip_pool_temporal', False)
                and hasattr(self, 'pool_log_sigma')):
            feats = self._temporal_refine_clips(feats)

        if pool == 'mean':
            return logits.mean(1), feats.mean(1)
        if pool == 'max':
            return logits.amax(dim=1), feats.amax(dim=1)
        if pool == 'attn':
            # Reduce per-clip feature to a 1-D summary [B, K, D] for scalar scoring.
            # For 4D+ features (e.g. CBraMod [B,K,Ch,T,D]) mean across spatial dims keeping last D.
            if feats.dim() > 3:
                spatial_dims = tuple(range(2, feats.dim() - 1))
                score_input = feats.mean(dim=spatial_dims)
            else:
                score_input = feats  # already [B, K, D]
            attn_layer = self.clip_attn_v if modality == 'video' else self.clip_attn_e
            scores = attn_layer(score_input).squeeze(-1)            # [B, K]
            w = torch.softmax(scores, dim=1)                        # [B, K]
            w_feat  = w.view((B, K) + (1,) * (feats.dim() - 2))
            w_logit = w.view((B, K) + (1,) * (logits.dim() - 2))
            feat  = (feats  * w_feat ).sum(1)
            logit = (logits * w_logit).sum(1)
            return logit, feat
        raise ValueError(f"Unknown clip_pool: {pool}")

    @staticmethod
    def _sinusoid_pe(K, D, device):
        """Standard transformer sinusoidal positional embedding.

        Returns [K, D] tensor. Parameter-free and handles variable K.
        Used on clip nodes (when --gcn_clip_pe) so GCN can distinguish
        temporal positions of dense-N clips.
        """
        pos = torch.arange(K, device=device, dtype=torch.float32).unsqueeze(1)  # [K, 1]
        div = torch.exp(
            torch.arange(0, D, 2, device=device, dtype=torch.float32)
            * -(math.log(10000.0) / D)
        )  # [D/2]
        pe = torch.zeros(K, D, device=device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def setup_dense_cache(self, cache_dir):
        """Enable on-disk feature cache shared across DDP ranks and epochs.

        Should be called once after model construction (e.g. from runner.py)
        with a path that all ranks can read/write. Pass None or '' to disable.
        """
        import os
        if not cache_dir:
            self._video_block_cache_dir = None
            return
        os.makedirs(cache_dir, exist_ok=True)
        self._video_block_cache_dir = cache_dir

    def _cache_path(self, sample_idx):
        import os
        return os.path.join(self._video_block_cache_dir, f'{int(sample_idx)}.pt')

    def _load_cached_block(self, sample_idx):
        """Return cached fp16 CPU tensor [N_actual, P, D] or None if absent.

        When disk cache is enabled, we DO NOT mirror loads into the in-memory
        dict — relying on OS page cache for fast subsequent disk reads. This
        prevents unbounded growth of self._video_block_cache (which over many
        epochs and DDP reshuffles ends up holding the full dataset per rank,
        contributing to slow CPU RAM exhaustion). With disk disabled, falls
        back to the memory dict only.
        """
        sid = int(sample_idx)
        cached = self._video_block_cache.get(sid, None)
        if cached is not None:
            return cached
        if self._video_block_cache_dir is None:
            return None
        import os
        path = self._cache_path(sid)
        if not os.path.exists(path):
            return None
        try:
            tensor = torch.load(path, map_location='cpu')
        except Exception:
            # Corrupt / partial file (e.g. crash mid-write). Treat as miss.
            return None
        # NOTE: no mirror to memory dict here — disk hits stay disk-only.
        # OS page cache makes repeat reads fast without unbounded RAM growth.
        return tensor

    def _save_cached_block(self, sample_idx, tensor):
        """Save fp16 CPU tensor to disk (primary) or in-memory dict (fallback).

        Disk path uses pid-suffixed tmp + atomic rename so concurrent writers
        from different ranks don't corrupt the final file. Skips disk write
        when destination already exists (another rank beat us to it).
        When disk write fails (disk full / permission / partial write) or no
        cache_dir was set, falls back to a bounded in-memory dict so training
        never crashes on cache IO. With disk enabled the in-memory dict stays
        empty — disk + OS page cache handles everything, bounding RAM growth.
        """
        sid = int(sample_idx)
        if self._video_block_cache_dir is None:
            # Memory-only fallback (no disk dir configured)
            self._video_block_cache[sid] = tensor
            return
        import os
        path = self._cache_path(sid)
        if os.path.exists(path):
            return
        tmp = path + f'.tmp.{os.getpid()}'
        try:
            torch.save(tensor, tmp)
            os.replace(tmp, path)
        except Exception as e:
            if os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
            # Disk write failed — keep this sample in memory only.
            self._video_block_cache[sid] = tensor
            if not getattr(self, '_dense_cache_warned', False):
                print(f'[dense_cache_features][WARN] disk write failed for sid={sid}: {e}. '
                      f'Falling back to in-memory cache for failed writes only.')
                self._dense_cache_warned = True

    def _dense_video_forward(self, video, lengths, chunk_size=6, sample_idxs=None, return_per_clip=False):
        """Dense N-clip video forward with memory-safe chunking and optional cache.

        Args:
            video: [B, N_max, C, T, H, W] padded clips (already transposed for backbone).
            lengths: [B] long tensor of actual N_i per sample.
            chunk_size: clips per inner forward.
            sample_idxs: [B] long tensor of dataset indices for feature caching.
            return_per_clip: when True, also return the un-pooled [B, N_max, D]
                per-clip feature tensor. Used by --gcn_video_per_clip with dense
                to feed per-clip nodes into the GCN region graph.

        Returns:
            (pooled_logit, pooled_feat) by default, or
            (pooled_logit, pooled_feat, per_clip_feat) when return_per_clip=True.
        """
        from torch.utils.checkpoint import checkpoint
        from tqdm import tqdm
        import torch.distributed as dist

        B, N_max = video.shape[:2]
        flat = video.reshape(B * N_max, *video.shape[2:])
        total = B * N_max
        chunk_size = max(1, int(chunk_size))

        _is_main = (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)

        # Determine block split for caching.
        n_trainable = max(0, int(getattr(self.args, 'video_unfreeze_last_n_blocks', 0)))
        has_split_api = (hasattr(self.video_model, 'forward_features_until')
                         and hasattr(self.video_model, 'forward_features_from'))
        cache_enabled = (
            getattr(self.args, 'dense_cache_features', False)
            and sample_idxs is not None
            and has_split_api
            and n_trainable > 0
        )
        n_blocks = len(getattr(self.video_model, 'blocks', [])) if has_split_api else 0
        split = (n_blocks - n_trainable) if has_split_api else 0

        if cache_enabled and split > 0:
            return self._dense_video_forward_cached(
                flat, lengths, sample_idxs, B, N_max, chunk_size, split, _is_main,
                return_per_clip=return_per_clip,
            )

        # --- Fallback: full forward without caching (existing behavior) ---
        def _fwd(c):
            return self.video_model(c, return_feat=True)

        _iters = range(0, total, chunk_size)
        if _is_main and total > chunk_size:
            _iters = tqdm(_iters, total=(total + chunk_size - 1) // chunk_size,
                          desc=f'dense fwd (N={N_max}, B={B})',
                          leave=False, mininterval=1.0)

        feat_chunks = []
        logit_chunks = []
        for i in _iters:
            chunk = flat[i:i + chunk_size]
            if self.training:
                logit, feat = checkpoint(_fwd, chunk, use_reentrant=False)
            else:
                with torch.no_grad():
                    logit, feat = _fwd(chunk)
            feat_chunks.append(feat)
            logit_chunks.append(logit)

        feat = torch.cat(feat_chunks, dim=0).view(B, N_max, -1)
        logit = torch.cat(logit_chunks, dim=0).view(B, N_max, -1)
        pooled_logit, pooled_feat = self._mask_pool(feat, logit, lengths, N_max)
        if return_per_clip:
            return pooled_logit, pooled_feat, feat
        return pooled_logit, pooled_feat

    def _mask_pool(self, feat, logit, lengths, N_max):
        """Pool over the N axis with mask. Honors --clip_pool ∈ {mean, max, attn}.

        mean (default): average of valid (non-padded) clips
        max:            per-feature-dim max over valid clips
        attn:           learned per-clip score (self.clip_attn_v) → softmax over
                        valid clips → weighted sum. Padded clips are masked to
                        -inf before softmax so they contribute zero weight.
        """
        device = feat.device
        mask = (torch.arange(N_max, device=device).unsqueeze(0)
                < lengths.to(device).unsqueeze(1)).float()  # [B, N_max]

        # Optional temporal-aware refinement of clip features BEFORE pool
        # (mirrors the GCN temporal method, applied to dense N-clip path).
        # Refine the masked feat; padded positions stay near-zero post-refine
        # because their attention rows are zeroed by the *= mask below.
        if (getattr(self.args, 'clip_pool_temporal', False)
                and hasattr(self, 'pool_log_sigma')
                and feat.dim() == 3 and feat.size(1) >= 2):
            feat_for_refine = feat * mask.unsqueeze(-1)  # zero padded clips
            refined = self._temporal_refine_clips(feat_for_refine)
            feat = refined * mask.unsqueeze(-1)          # keep padded zeroed

        pool_mode = getattr(self.args, 'clip_pool', 'mean')

        if pool_mode == 'attn' and hasattr(self, 'clip_attn_v'):
            # Score from feature; temperature sharpens softmax for more peaked weights.
            scores = self.clip_attn_v(feat).squeeze(-1)  # [B, N_max]
            _temp = float(getattr(self.args, 'clip_attn_temp', 1.0))
            if _temp > 0 and _temp != 1.0:
                scores = scores / _temp
            scores = scores.masked_fill(mask == 0, float('-inf'))
            w = torch.softmax(scores, dim=1)              # [B, N_max]
            # Safety: if a sample has 0 valid clips (shouldn't happen, but defensive),
            # softmax of all -inf returns NaN; replace with uniform over the row.
            w = torch.nan_to_num(w, nan=0.0)
            # One-time diagnostic (rank-0 only) so users can confirm the attn path
            # is actually firing AND that softmax weights are non-uniform. Also
            # reports raw score std so init/temperature can be tuned.
            if not getattr(self, '_mask_pool_attn_logged', False):
                import torch.distributed as _dist
                _is_main = (not _dist.is_available()) or (not _dist.is_initialized()) or (_dist.get_rank() == 0)
                if _is_main:
                    with torch.no_grad():
                        _w0 = w[0].detach().cpu()
                        _scores0 = scores[0].detach().float().cpu()
                        _valid_mask = mask[0].cpu().bool()
                        _n_valid = int(_valid_mask.sum().item())
                        _w_valid = _w0[_valid_mask]
                        _s_valid = _scores0[_valid_mask]
                        _uniform = 1.0 / max(1, _n_valid)
                        _dev = float((_w_valid - _uniform).abs().mean().item())
                        _s_std = float(_s_valid.std().item()) if _n_valid > 1 else 0.0
                    print(f'[attn_pool] active. sample0: N_valid={_n_valid}, '
                          f'temp={_temp}, score_std={_s_std:.4e}, '
                          f'w[0:3]={_w_valid[:3].tolist()}, '
                          f'mean_dev_from_uniform={_dev:.4e}')
                self._mask_pool_attn_logged = True
            w_u = w.unsqueeze(-1)
            return (logit * w_u).sum(dim=1), (feat * w_u).sum(dim=1)

        if pool_mode == 'max':
            # Make padded positions very negative so they lose the max.
            neg_inf = torch.finfo(feat.dtype).min
            feat_m = feat.masked_fill(mask.unsqueeze(-1) == 0, neg_inf)
            logit_m = logit.masked_fill(mask.unsqueeze(-1) == 0, neg_inf)
            return logit_m.amax(dim=1), feat_m.amax(dim=1)

        # Default: mean
        if not getattr(self, '_mask_pool_mean_logged', False):
            import torch.distributed as _dist
            _is_main = (not _dist.is_available()) or (not _dist.is_initialized()) or (_dist.get_rank() == 0)
            if _is_main:
                _n_valid = int(mask[0].sum().item())
                print(f'[mean_pool] active. sample0 N_valid={_n_valid}, '
                      f'uniform_w=1/{_n_valid}={1.0/max(1, _n_valid):.4e}')
            self._mask_pool_mean_logged = True
        mask_u = mask.unsqueeze(-1)
        counts = lengths.to(device).float().clamp_min(1).unsqueeze(-1)
        return (logit * mask_u).sum(dim=1) / counts, (feat * mask_u).sum(dim=1) / counts

    def _dense_video_forward_cached(self, flat, lengths, sample_idxs, B, N_max, chunk_size, split, is_main, return_per_clip=False):
        """Cached two-stage dense forward.

        Stage 1 (frozen blocks 0..split-1): for any sample not yet cached, run
        no_grad forward to produce the block-split intermediate, then save to
        CPU cache keyed by dataset idx. Cached samples are loaded directly.
        Stage 2 (trainable blocks split..end + norm + fc_norm + head): always
        runs through the trainable tail with gradient checkpointing.
        """
        from torch.utils.checkpoint import checkpoint
        from tqdm import tqdm

        device = flat.device

        # Indices in the [0, B) batch axis that need a stage-1 forward.
        miss_idx_in_batch = []
        miss_sample_ids = []
        cached_inter_slots = [None] * B  # per-sample tensor [N_max, P, D] on device
        sids_list = sample_idxs.tolist()

        for b in range(B):
            sid = sids_list[b]
            cached = self._load_cached_block(sid)
            if cached is None:
                miss_idx_in_batch.append(b)
                miss_sample_ids.append(sid)
                continue
            # Cached tensor is [N_actual, P, D] on CPU. Cycle-pad to N_max to match.
            N_i = int(lengths[b].item())
            if cached.size(0) != N_i:
                # Defensive: cache shape mismatch → drop and re-extract.
                self._video_block_cache.pop(sid, None)
                miss_idx_in_batch.append(b)
                miss_sample_ids.append(sid)
                continue
            if N_i < N_max:
                n_rep = (N_max + N_i - 1) // N_i
                padded = cached.repeat((n_rep,) + (1,) * (cached.dim() - 1))[:N_max]
            else:
                padded = cached[:N_max]
            # Restore fp32 on GPU (cache stored as fp16 for memory).
            cached_inter_slots[b] = padded.to(device=device, dtype=torch.float32, non_blocking=True)

        # Stage 1 for cache misses (no_grad, chunked).
        if miss_idx_in_batch:
            miss_b_t = torch.tensor(miss_idx_in_batch, device=device, dtype=torch.long)
            # Indices into the flat [B*N_max] tensor for missing samples.
            offsets = miss_b_t.unsqueeze(1) * N_max + torch.arange(N_max, device=device).unsqueeze(0)
            offsets = offsets.reshape(-1)  # [len(miss)*N_max]
            miss_flat = flat[offsets]  # [len(miss)*N_max, C, T, H, W]
            miss_total = miss_flat.size(0)

            _iters_s1 = range(0, miss_total, chunk_size)
            if is_main and miss_total > chunk_size:
                _iters_s1 = tqdm(_iters_s1, total=(miss_total + chunk_size - 1) // chunk_size,
                                 desc=f'dense stage1 [miss={len(miss_idx_in_batch)}]',
                                 leave=False, mininterval=1.0)

            inter_chunks = []
            with torch.no_grad():
                for i in _iters_s1:
                    c = miss_flat[i:i + chunk_size]
                    h = self.video_model.forward_features_until(c, split)
                    inter_chunks.append(h)
            inter_miss = torch.cat(inter_chunks, dim=0)  # [len(miss)*N_max, P, D]
            inter_miss = inter_miss.view(len(miss_idx_in_batch), N_max, *inter_miss.shape[1:])

            for j, b in enumerate(miss_idx_in_batch):
                cached_inter_slots[b] = inter_miss[j]  # [N_max, P, D] on device
                # Save the actual N_i clips (not cycle-padded) as fp16 CPU; the
                # helper writes to in-memory + disk (if cache_dir set).
                N_i = int(lengths[b].item())
                self._save_cached_block(
                    miss_sample_ids[j],
                    inter_miss[j, :N_i].detach().to(dtype=torch.float16).cpu(),
                )

        # Assemble stage-1 output for the full batch.
        inter_full = torch.stack(cached_inter_slots, dim=0)  # [B, N_max, P, D]
        inter_flat = inter_full.reshape(B * N_max, *inter_full.shape[2:])
        total = B * N_max

        # Stage 2: trainable tail. Re-enable grad on the input so autograd tracks.
        inter_flat = inter_flat.detach().requires_grad_(self.training)

        def _tail_fwd(h_chunk):
            global_f = self.video_model.forward_features_from(h_chunk, split)
            # AdaMAE/VideoMAE only build self.head when set_video_only or fusion='router';
            # for eeg_signal + non-router fusions head is not present. Mirror the existing
            # _run_video_backbone fallback: return global_f as the logit placeholder so
            # callers that don't actually use video_logit (e.g. naive concat / GCN-only)
            # still work, while set_video_only / router get the real head output.
            head = getattr(self.video_model, 'head', None)
            if head is not None:
                # AdaMAE has fc_dropout, VideoMAE has head_dropout — accept either.
                dropout = (getattr(self.video_model, 'fc_dropout', None)
                           or getattr(self.video_model, 'head_dropout', None))
                feat_for_head = dropout(global_f) if dropout is not None else global_f
                logit = head(feat_for_head)
            else:
                logit = global_f
            return logit, global_f

        _iters_s2 = range(0, total, chunk_size)
        if is_main and total > chunk_size:
            _iters_s2 = tqdm(_iters_s2, total=(total + chunk_size - 1) // chunk_size,
                             desc='dense stage2 (tail)',
                             leave=False, mininterval=1.0)

        feat_chunks = []
        logit_chunks = []
        for i in _iters_s2:
            h_chunk = inter_flat[i:i + chunk_size]
            if self.training:
                logit, feat = checkpoint(_tail_fwd, h_chunk, use_reentrant=False)
            else:
                with torch.no_grad():
                    logit, feat = _tail_fwd(h_chunk)
            feat_chunks.append(feat)
            logit_chunks.append(logit)

        feat = torch.cat(feat_chunks, dim=0).view(B, N_max, -1)
        logit = torch.cat(logit_chunks, dim=0).view(B, N_max, -1)
        pooled_logit, pooled_feat = self._mask_pool(feat, logit, lengths, N_max)
        if return_per_clip:
            return pooled_logit, pooled_feat, feat
        return pooled_logit, pooled_feat

    def _run_video_backbone(self, video, return_feat=False, return_tokens=False):
        """Run video backbone, handling K-clip input [B,K,C,T,H,W] transparently.

        For K>1 clips: run backbone on [B*K, C, T, H, W], pool features and logits
        across K via the chosen clip_pool method (mean/max/attn).
        """
        _multi = (video.dim() == 6)
        if not _multi:
            return self.video_model(video, return_feat=return_feat, return_tokens=return_tokens)
        B, K = video.shape[:2]
        flat = video.view(B * K, *video.shape[2:])
        logit_flat, feat_flat = self.video_model(flat, return_feat=True)
        logit, feat = self._pool_clips(feat_flat, logit_flat, B, K, modality='video')
        if return_feat:
            return logit, feat
        return logit

    def _run_eeg_backbone(self, x, return_per_clip=False, return_patches=False):
        """Run EEG backbone, handling K-clip input transparently.

        CBraMod input is already 4D per batch ([B, Ch, T_seg, seg_len]) at K=1.
        K-clip mode bumps "eeg"/"eeg_local" to 5D ([B, K, Ch, T_seg, seg_len]); when
        --eeg_full_signal is set, EEG stays 4D even for K>1 (single full-view window).
        Other keys (video, output, ...) are left untouched.

        Args:
            return_per_clip: if True, additionally return per-clip features
                [B, K, D] (un-pooled across K). For 4D input (K=1), per-clip
                degenerates to feat.unsqueeze(1) of shape [B, 1, D].
            return_patches: if True, additionally return CBraMod's encoder
                temporal patches as [B, T_seq, D_eeg=200] for cross-modal
                attention. 4D input → T_seq = T_seg (e.g. 10). 5D K-clip →
                T_seq = K × T_seg (flattened along time so video clips can
                attend to fine-grained temporal positions).
        """
        _multi = (x["eeg"].dim() == 5)

        if not _multi:
            if return_patches:
                logit, feat, enc_feats = self.eeg_model(x, return_encoder_feats=True)
                # enc_feats: [B, ch, T_seg, D] → mean over channels
                patches = enc_feats.mean(dim=1)  # [B, T_seg, D]
            else:
                logit, feat = self.eeg_model(x)
                patches = None

            results = [logit, feat]
            if return_per_clip:
                results.append(feat.unsqueeze(1))  # [B, 1, D]
            if return_patches:
                results.append(patches)
            return tuple(results)

        B, K = x["eeg"].shape[:2]
        x_flat = dict(x)
        for key in ("eeg", "eeg_local"):
            if key in x and torch.is_tensor(x[key]):
                v = x[key]
                x_flat[key] = v.view(B * K, *v.shape[2:])

        if return_patches:
            logit_flat, feat_flat, enc_feats_flat = self.eeg_model(x_flat, return_encoder_feats=True)
        else:
            logit_flat, feat_flat = self.eeg_model(x_flat)

        logit, feat = self._pool_clips(feat_flat, logit_flat, B, K, modality='eeg')

        results = [logit, feat]
        if return_per_clip:
            results.append(feat_flat.view(B, K, -1))  # [B, K, D]
        if return_patches:
            # enc_feats_flat: [B*K, ch, T_seg, D] → mean ch → [B*K, T_seg, D]
            # then flatten K into time: [B, K*T_seg, D]
            ch_pooled = enc_feats_flat.mean(dim=1)
            T_seg = ch_pooled.size(1)
            D_eeg = ch_pooled.size(2)
            patches = ch_pooled.view(B, K * T_seg, D_eeg)
            results.append(patches)
        return tuple(results)

    def forward(self, x):
        eeg = x["eeg"]
        video = x["video"].transpose_(-3, -4)
        eeg_stft = x["eeg_local"]

        # Dense N-clip video mode: presence of 'video_lengths' (added by
        # dense_video_collate_fn) is the unambiguous activation signal.
        # Forces pooled video global → standard pooled GCN; not compatible
        # with --gcn_*_per_clip or --cross_modal_temporal_attn.
        video_lengths = x.get("video_lengths", None)
        sample_idxs = x.get("sample_idx", None)  # for cross-epoch frozen-feature cache
        _dense_active = (video_lengths is not None)
        _dense_chunk = int(getattr(self.args, 'dense_chunk_size', 6))

        if self.args.set_eeg_only:
            # --eeg_full_signal=True: single full-view 4D EEG window → CBraMod 직접.
            # --eeg_full_signal=False: K-clip per-clip coupled EEG. K=1이면 4D, K>1이면
            # 5D [B, K, Ch, T_seg, P] → _run_eeg_backbone가 flatten+pool 처리
            # (set_video_only의 K-clip 경로 미러).
            if not getattr(self.args, 'eeg_full_signal', False):
                output = self._run_eeg_backbone(x)
            else:
                output = self.eeg_model(x)

        elif self.args.set_video_only and not self.args.eeg_signal:
            per_clip_logits = None  # set only when per_clip_aux_loss is enabled
            if _dense_active:
                # Dense N-clip forward → single pooled video logit.
                output_pool, _ = self._dense_video_forward(
                    video, video_lengths, chunk_size=_dense_chunk,
                    sample_idxs=sample_idxs,
                )
                if self.args.dataset in ('emognition', 'mdmer'):
                    x_v = output_pool[:, :self.output_dim[0]].unsqueeze(-1)
                    x_a = output_pool[:, self.output_dim[0]:].unsqueeze(-1)
                    output = torch.cat((x_v, x_a), dim=-1)
                else:
                    output = output_pool
            elif video.dim() == 6:
                B, K = video.shape[:2]
                flat = video.view(B * K, *video.shape[2:])
                # Need features for attn-pool; mean/max also work on (logit_flat, feat_flat).
                logit_flat, feat_flat = self.video_model(flat, return_feat=True)
                output_pool, _ = self._pool_clips(feat_flat, logit_flat, B, K, modality='video')
                # video_model keeps logits flat ([B*K, n_v+n_a]) when num_clips>1
                # (see modeling_finetune_v0 forward); apply V/A split externally here.
                if self.args.dataset in ('emognition', 'mdmer'):
                    x_v = output_pool[:, :self.output_dim[0]].unsqueeze(-1)
                    x_a = output_pool[:, self.output_dim[0]:].unsqueeze(-1)
                    output = torch.cat((x_v, x_a), dim=-1)
                    if getattr(self.args, 'per_clip_aux_loss', 0.0) > 0:
                        # logit_flat: [B*K, n_v + n_a]  →  [B, K, n_v, 2]
                        lf = logit_flat.view(B, K, -1)
                        pc_v = lf[..., :self.output_dim[0]].unsqueeze(-1)
                        pc_a = lf[..., self.output_dim[0]:].unsqueeze(-1)
                        per_clip_logits = torch.cat((pc_v, pc_a), dim=-1)  # [B, K, n_v, 2]
                else:
                    output = output_pool
                    if getattr(self.args, 'per_clip_aux_loss', 0.0) > 0:
                        per_clip_logits = logit_flat.view(B, K, -1)        # [B, K, n_classes]
            else:
                output = self.video_model(video)
                # Single-clip path: backbones gate their internal V/A split on
                # args.num_clips==1. With --train_random_crop + num_clips>1 (val
                # uses K-clip), training feeds a single clip while num_clips=K,
                # so the head leaves logits flat [B, n_v+n_a]. Reshape here so
                # the trainer's [B, n_v, 2] CE path applies uniformly.
                if (
                    output.dim() == 2
                    and self.args.dataset in ('emognition', 'mdmer')
                    and output.size(-1) == self.output_dim[0] * self.output_dim[1]
                ):
                    x_v = output[:, :self.output_dim[0]].unsqueeze(-1)
                    x_a = output[:, self.output_dim[0]:].unsqueeze(-1)
                    output = torch.cat((x_v, x_a), dim=-1)

            if per_clip_logits is not None:
                return output, per_clip_logits

        elif self.args.eeg_signal:
            # Dense N-clip video → single pooled video global (mask-pool over N).
            # Forces pooled GCN — disables per-clip / cross-modal temporal options
            # which assume fixed-K video clip nodes / explicit alignment.
            # Cross-modal temporal attention: per-clip video Q over EEG patches K/V,
            # aligned per-clip features then K-pool to a single video global.
            # Mutually exclusive with per-clip GCN flags — when this is active, the
            # graph stays 4-node (pooled), avoiding per-clip over-smoothing.
            _use_xmod_temporal = (
                getattr(self.args, 'cross_modal_temporal_attn', False)
                and getattr(self.args, 'gcn', False)
                and (video.dim() == 6)
                and (not _dense_active)
            )

            # Per-clip GCN node activation flags. Supported when video has a
            # multi-clip dim (6D K-clip OR dense N-clip), EEG has K-clip 5D, and
            # cross-modal temporal attention is NOT active.
            _use_per_clip = (
                getattr(self.args, 'gcn_video_per_clip', False)
                and getattr(self.args, 'gcn', False)
                and (video.dim() == 6 or _dense_active)
                and (not _use_xmod_temporal)
            )
            _use_eeg_per_clip = (
                getattr(self.args, 'gcn_eeg_per_clip', False)
                and getattr(self.args, 'gcn', False)
                and (x["eeg"].dim() == 5)
                and (not _use_xmod_temporal)
            )

            video_clips = None  # [B, K_v, D] when per-clip video is active
            eeg_clips = None    # [B, K_e, D] when per-clip EEG is active
            eeg_patches = None  # [B, T_seq, D_eeg=200] when xmod temporal is active

            if _use_xmod_temporal:
                eeg_logit, eeg_f, eeg_patches = self._run_eeg_backbone(x, return_patches=True)
            elif _use_eeg_per_clip:
                eeg_logit, eeg_f, eeg_clips = self._run_eeg_backbone(x, return_per_clip=True)
            else:
                eeg_logit, eeg_f = self._run_eeg_backbone(x)

            if _dense_active:
                # Dense N-clip → mask-pooled video global. When --gcn_video_per_clip
                # is on, also expose the un-pooled per-clip features as graph nodes;
                # otherwise the downstream GCN block treats it as 4-node graph.
                if _use_per_clip:
                    video_logit, video_f, video_clips = self._dense_video_forward(
                        video, video_lengths, chunk_size=_dense_chunk,
                        sample_idxs=sample_idxs, return_per_clip=True,
                    )
                else:
                    video_logit, video_f = self._dense_video_forward(
                        video, video_lengths, chunk_size=_dense_chunk,
                        sample_idxs=sample_idxs,
                    )
            elif getattr(self.args, 'eeg_video_spatial_attn', False):
                video_logit, video_f, video_tokens = self.video_model(
                    video, return_feat=True, return_tokens=True
                )
            elif _use_xmod_temporal:
                # Per-clip video → cross-attend over EEG patches → K-pool to global.
                B_v, K_v = video.shape[:2]
                flat = video.view(B_v * K_v, *video.shape[2:])
                logit_flat, feat_flat = self.video_model(flat, return_feat=True)
                video_logit, _ = self._pool_clips(
                    feat_flat, logit_flat, B_v, K_v, modality='video'
                )
                video_clips_raw = feat_flat.view(B_v, K_v, -1)  # [B, K, 768]

                # Cross-attn: Q=video clips, K/V=EEG patches.
                aligned, _attn_w = self.cross_modal_temporal_attn_module(
                    video_clips_raw, eeg_patches, eeg_patches
                )
                alpha = torch.sigmoid(self.cross_modal_temporal_alpha)
                video_clips_aligned = video_clips_raw + alpha * aligned  # [B, K, 768]

                # K-pool aligned clips → single video global for pooled GCN.
                video_f = video_clips_aligned.mean(dim=1)
            elif _use_per_clip:
                B_v, K_v = video.shape[:2]
                flat = video.view(B_v * K_v, *video.shape[2:])
                logit_flat, feat_flat = self.video_model(flat, return_feat=True)
                video_logit, video_f = self._pool_clips(
                    feat_flat, logit_flat, B_v, K_v, modality='video'
                )
                video_clips = feat_flat.view(B_v, K_v, -1)  # [B, K, D] for graph
            else:
                video_logit, video_f = self._run_video_backbone(video, return_feat=True)

            logit_eeg = eeg_logit
            logit_video = video_logit

            # EEG temporal cross-attention (asymmetric Q=video, K/V=EEG segments)
            if getattr(self.args, 'eeg_temporal_attn', False):
                eeg_seq = eeg_f.mean(dim=1)                  # [B, 10, D]
                q = video_f.unsqueeze(1)                          # [B, 1, D]
                eeg_t, _ = self.eeg_temporal_attn_module(q, eeg_seq, eeg_seq)
                eeg_t = eeg_t.squeeze(1)                          # [B, D]
                alpha = torch.sigmoid(self.eeg_temporal_alpha)
                eeg_f = eeg_f + alpha * eeg_t

            # eeg_feat = eeg_embed.mean(dim=1)
            # eeg_feat = self.eeg_proj(eeg_feat)
            # video_f = self.video_proj(video_f)

            if self.args.fusion == 'attention':
                video_f = self.cross_attn(eeg_f, video_f)

            elif self.args.fusion == 'FiLM':
                video_f = self.film(video_f, eeg_f)

            fused_f = torch.cat((eeg_f, video_f), dim=1)

            if self.args.gcn:
                # B, ch, _, _ = eeg_embed.shape
                # eeg_flat = eeg_embed.mean(dim=2)  # [B, ch, 200]

                # B, ch, _, _ = eeg_raw.shape
                # eeg_flat = eeg_raw.view(B, ch, -1)

                if getattr(self.args, 'ch_temporal_attn', False):
                    # Channel-level temporal attention: video_f guides each channel's
                    # temporal focus over CBraMod encoder output [B, Ch, 10, 200]
                    B_ct, Ch_ct, T_ct, D_ct = eeg_f.shape
                    _eeg_kv_src = eeg_f.detach() if getattr(self.args, 'stop_gradient', False) else eeg_f
                    _vid_q_src  = video_f.detach()   if getattr(self.args, 'stop_gradient', False) else video_f
                    eeg_kv = _eeg_kv_src.reshape(B_ct * Ch_ct, T_ct, D_ct)           # [B*Ch, 10, 200]
                    video_q = _vid_q_src.unsqueeze(1).expand(-1, Ch_ct, -1).reshape(B_ct * Ch_ct, 1, 768)  # [B*Ch, 1, 768]
                    eeg_local_t, _ = self.ch_temporal_attn_module(video_q, eeg_kv, eeg_kv)             # [B*Ch, 1, 768]
                    eeg_local = eeg_local_t.squeeze(1).view(B_ct, Ch_ct, 768)                           # [B, Ch, 768]
                    ch = Ch_ct
                elif getattr(self.args, 'gcn_region_eeg_source', 'stft') == 'cbramod':
                    # Per-channel features for GCN region nodes.
                    # Shape contract: [B, ch, T_seg=10, 200] regardless of backbone.
                    # CBraMod: pre-positional features from PatchEmbedding (channel-
                    #   independent, BEFORE the 19-ch positional conv).
                    # REVE:    REVE doesn't expose a documented pre-positional split,
                    #   so we use return_per_channel_pre=True which returns the
                    #   post-encoder per-channel features (already projected to dim 200
                    #   and pooled to T_seg=10 in the REVE wrapper). Shape-identical
                    #   so downstream eeg_feat_proj works unchanged.
                    pe_input = x["eeg"]
                    _is_reve = getattr(self.args, 'eeg_backbone', 'cbramod') == 'reve'
                    if _is_reve:
                        # REVE: full forward (no patch_embedding shortcut). Cost is
                        # one extra REVE pass per step; acceptable since EEG is the
                        # cheaper modality here. If perf becomes an issue, cache the
                        # output during the main EEG forward and reuse.
                        if pe_input.dim() == 5:
                            _B, _K = pe_input.shape[:2]
                            pe_in_flat = pe_input.view(_B * _K, *pe_input.shape[2:])
                            _, _, eeg_pre = self.eeg_model({"eeg": pe_in_flat}, return_per_channel_pre=True)
                            eeg_pre = eeg_pre.view(_B, _K, *eeg_pre.shape[1:]).mean(dim=1)
                        else:
                            _, _, eeg_pre = self.eeg_model({"eeg": pe_input}, return_per_channel_pre=True)
                    else:
                        if pe_input.dim() == 5:
                            _B, _K = pe_input.shape[:2]
                            pe_in_flat = pe_input.view(_B * _K, *pe_input.shape[2:])
                            _, _, eeg_pre = self.eeg_model.patch_embedding(pe_in_flat, return_pre_pos=True)
                            eeg_pre = eeg_pre.view(_B, _K, *eeg_pre.shape[1:]).mean(dim=1)
                        else:
                            _, _, eeg_pre = self.eeg_model.patch_embedding(pe_input, return_pre_pos=True)
                    # eeg_pre: [B, ch, T_seg, 200]
                    # Detach by default to isolate the GCN-region path from CBraMod's
                    # patch_embedding parameters.
                    _no_detach = getattr(self.args, 'gcn_region_eeg_no_detach', False)
                    _proj_kind = getattr(self.args, 'gcn_region_eeg_proj', 'linear')
                    if _proj_kind == 'conv':
                        # Conv1d path — keep T_seg dim, learn temporal patterns per channel.
                        eeg_in = eeg_pre if _no_detach else eeg_pre.detach()
                        _B, _ch, _T, _D = eeg_in.shape
                        # Conv1d expects [N, C, L]; here N = B*ch, C = 200, L = T_seg
                        eeg_in_flat = eeg_in.permute(0, 1, 3, 2).reshape(_B * _ch, _D, _T)
                        eeg_out_flat = self.eeg_feat_proj(eeg_in_flat).squeeze(-1)  # [B*ch, 768]
                        eeg_local = eeg_out_flat.view(_B, _ch, -1)                  # [B, ch, 768]
                    else:
                        # Linear path — mean over T_seg → Linear(200, 768)
                        eeg_per_ch = eeg_pre.mean(dim=2)
                        if not _no_detach:
                            eeg_per_ch = eeg_per_ch.detach()
                        eeg_local = self.eeg_feat_proj(eeg_per_ch)
                    ch = eeg_local.size(1)
                else:
                    _stft = eeg_stft.mean(dim=1) if eeg_stft.dim() == 5 else eeg_stft
                    B, ch, _, _ = _stft.shape
                    eeg_flat = _stft.view(B, ch, -1)
                    eeg_local = self.eeg_feat_proj(eeg_flat)  # [B, ch, D]

                if self.args.dataset == "eav":
                    region_dataset = [
                        [0,1,2,3,4,5,6,7,8,9],      # Frontal (Fp1, Fp2, F7, F3, Fz, F4, F8, FC5, FC1, FC2)
                        [10,14],                    # Temporal (T7, T8)
                        [11,12,13,15,16,17],        # Central (C3, Cz, C4, FC6, CP5, CP1)
                        [18,19,20,21,22,23,24],     # Parietal (CP2, CP6, P7, P3, Pz, P4, P8)
                        [25,26,27,28,29]            # Occipital (PO9, O1, Oz, O2, PO10)
                    ]
                elif self.args.dataset == "emognition":
                    region_dataset = [
                    [1, 2],  # Frontal
                    [0, 3],] # Temporal
                elif self.args.dataset == "mdmer":
                    region_dataset = [
                    [0,1,2,3,4,5,6],      # Frontal
                    [7,11],               # Temporal
                    [8,9,10],             # Central
                    [12,13,14],           # Parietal
                    [15,16,17],]          # Occipital

                if self.args.gcn_step1:
                    eeg_gcn = self.gcn_local(eeg_local, region_indices = region_dataset)
                else:
                    eeg_gcn = self.gcn_local(eeg_local, region_indices = None)

                eeg_region = self.region_pool(eeg_gcn, region_indices = region_dataset)

                # region-only: region node importance
                if self.use_region_importance:
                    eeg_region, region_alpha = self.apply_region_importance(eeg_region, video_f, eeg_f)

                # EEG-conditioned video spatial re-attention:
                if getattr(self.args, 'eeg_video_spatial_attn', False):
                    vid_reattend, _ = self.eeg_spatial_attn(
                        eeg_region.detach(), video_tokens, video_tokens
                    )  # [B, R, 768]
                    vid_f_spatial = vid_reattend.mean(dim=1)  # [B, 768]
                    alpha_s = torch.sigmoid(self.eeg_spatial_alpha)
                    video_f = video_f + alpha_s * vid_f_spatial

                if getattr(self.args, 'conf_gate', False):
                    logit_eeg = self.eeg_head(eeg_f)
                    logit_video = self.video_head(video_f)
                    H_e = -(F.softmax(logit_eeg, -1) * F.log_softmax(logit_eeg, -1)).sum(-1)
                    H_v = -(F.softmax(logit_video, -1) * F.log_softmax(logit_video, -1)).sum(-1)
                    w = torch.softmax(-torch.stack([H_v, H_e], dim=-1), dim=-1)
                    video_node = w[:, 0:1] * video_f
                    eeg_node   = w[:, 1:2] * eeg_f
                elif getattr(self.args, 'stop_gradient', False):
                    # native backbone logits are already computed above:
                    # eeg_logit, video_logit
                    video_node = video_f.detach()
                    eeg_node = eeg_f.detach()
                else:
                    video_node = video_f
                    eeg_node = eeg_f

                # Choose video-side node(s): K clip nodes when per-clip active,
                # else the single (gated/detached) video_node.
                if _use_per_clip and video_clips is not None:
                    if getattr(self.args, 'stop_gradient', False):
                        v_clip_nodes = video_clips.detach()
                    elif getattr(self.args, 'conf_gate', False):
                        v_clip_nodes = video_clips * w[:, 0:1].unsqueeze(1)
                    else:
                        v_clip_nodes = video_clips
                else:
                    v_clip_nodes = video_node.unsqueeze(1)  # [B, 1, D]
                num_video_nodes = v_clip_nodes.size(1)

                # Same treatment on EEG side.
                if _use_eeg_per_clip and eeg_clips is not None:
                    if getattr(self.args, 'stop_gradient', False):
                        e_clip_nodes = eeg_clips.detach()
                    elif getattr(self.args, 'conf_gate', False):
                        e_clip_nodes = eeg_clips * w[:, 1:2].unsqueeze(1)
                    else:
                        e_clip_nodes = eeg_clips
                else:
                    e_clip_nodes = eeg_node.unsqueeze(1)    # [B, 1, D]
                num_eeg_nodes = e_clip_nodes.size(1)

                # Optional: add sinusoidal temporal positional encoding to clip
                # nodes so the GCN can distinguish "which clip in time" each is.
                # Cost-free (no params), applied only when more than one clip node
                # on that modality side (single-pool case → no time axis).
                if getattr(self.args, 'gcn_clip_pe', False):
                    if num_video_nodes > 1:
                        v_clip_nodes = v_clip_nodes + self._sinusoid_pe(
                            num_video_nodes, v_clip_nodes.size(-1), v_clip_nodes.device,
                        ).unsqueeze(0)
                    if num_eeg_nodes > 1:
                        e_clip_nodes = e_clip_nodes + self._sinusoid_pe(
                            num_eeg_nodes, e_clip_nodes.size(-1), e_clip_nodes.device,
                        ).unsqueeze(0)

                all_nodes = torch.cat([
                    v_clip_nodes,    # [B, K_v, D]
                    e_clip_nodes,    # [B, K_e, D]
                    eeg_region,      # [B, R,   D]
                ], dim=1)

                # Per-node time position for the optional temporal-distance adjacency
                # prior. Video clips at [0..K_v-1], EEG clips at [0..K_e-1] (same
                # absolute time when K_v==K_e), region nodes at -1 (no time).
                _N = all_nodes.size(1)
                _node_t = torch.full((_N,), -1, dtype=torch.long, device=all_nodes.device)
                if num_video_nodes > 1:
                    _node_t[:num_video_nodes] = torch.arange(num_video_nodes, device=all_nodes.device)
                if num_eeg_nodes > 1:
                    _node_t[num_video_nodes:num_video_nodes + num_eeg_nodes] = torch.arange(
                        num_eeg_nodes, device=all_nodes.device,
                    )

                output = self.gcn_region(
                    all_nodes,
                    num_video_nodes=num_video_nodes,
                    num_eeg_nodes=num_eeg_nodes,
                    node_time_pos=_node_t,
                )

                if self.args.fusion == 'router':
                    logit_gcn = output

                    # logit_eeg, logit_video are native backbone classifier outputs
                    _use_va = hasattr(self, 'router_mlp_v')
                    _mix_fn = self._mix_logits_va if _use_va else self._mix_logits

                    if getattr(self.args, 'stop_gradient', False):
                        output, router_w, router_logits = _mix_fn(
                            logit_eeg.detach(),   # backbone → no gradient
                            logit_video.detach(), # backbone → no gradient
                            logit_gcn             # GCN is trainable → keep gradient
                        )
                    else:
                        output, router_w, router_logits = _mix_fn(logit_eeg, logit_video, logit_gcn)

                elif self.args.fusion == 'entropy_gate':
                    logit_gcn = output
                    logit_eeg = self.eeg_head(eeg_f)
                    logit_video = self.video_head(video_f)
                    H_eeg   = -(torch.softmax(logit_eeg,   dim=-1) * torch.log_softmax(logit_eeg,   dim=-1)).sum(-1)
                    H_video = -(torch.softmax(logit_video, dim=-1) * torch.log_softmax(logit_video, dim=-1)).sum(-1)
                    H_gcn   = -(torch.softmax(logit_gcn,   dim=-1) * torch.log_softmax(logit_gcn,   dim=-1)).sum(-1)
                    H_stack = torch.stack([H_eeg, H_video, H_gcn], dim=-1)  # [B, 3]
                    w = torch.softmax(-H_stack, dim=-1)                     # [B, 3]
                    output = w[:, 0:1] * logit_eeg + w[:, 1:2] * logit_video + w[:, 2:3] * logit_gcn

            elif self.args.fusion == 'router':
                # 2-way router (no GCN): EEG vs Video
                logit_eeg = eeg_logit
                logit_video = video_logit
                logit_gcn = torch.zeros_like(logit_eeg)
                _use_va = hasattr(self, 'router_mlp_v')
                if getattr(self.args, 'stop_gradient', False):
                    if _use_va:
                        output, router_w, router_logits = self._mix_logits_va_2way(
                            logit_eeg.detach(), logit_video.detach()
                        )
                    else:
                        output, router_w, router_logits = self._mix_logits_2way(
                            logit_eeg.detach(), logit_video.detach()
                        )
                else:
                    if _use_va:
                        output, router_w, router_logits = self._mix_logits_va_2way(logit_eeg, logit_video)
                    else:
                        output, router_w, router_logits = self._mix_logits_2way(logit_eeg, logit_video)
            else:
                fused_f = F.normalize(fused_f, dim=-1)
                output = self.classifier(fused_f)

            if self.args.dataset in ('emognition', 'mdmer'):

                # output: [B, 2C] -> [B, C, 2]
                if output.dim() == 2:
                    output_v = output[:, :self.output_dim[0]].unsqueeze(-1)
                    output_a = output[:, self.output_dim[0]:].unsqueeze(-1)
                    output = torch.cat((output_v, output_a), dim=-1)

                if self.args.fusion == 'router':
                    pass

                elif getattr(self.args, 'conf_gate', False) or getattr(self.args, 'stop_gradient', False):
                    logit_eeg_v = logit_eeg[:, :self.output_dim[0]].unsqueeze(-1)
                    logit_eeg_a = logit_eeg[:, self.output_dim[0]:].unsqueeze(-1)
                    logit_eeg = torch.concat((logit_eeg_v, logit_eeg_a), dim=-1)

                    logit_video_v = logit_video[:, :self.output_dim[0]].unsqueeze(-1)
                    logit_video_a = logit_video[:, self.output_dim[0]:].unsqueeze(-1)
                    logit_video = torch.concat((logit_video_v, logit_video_a), dim=-1)

        needs_logits = (
            getattr(self.args, 'conf_gate', False)
            or getattr(self.args, 'stop_gradient', False)
        )

        if self.args.fusion == 'router':
            return output, logit_gcn, logit_eeg, logit_video, router_logits

        elif needs_logits:
            if self.args.coral_loss:
                return output, logit_eeg, logit_video, eeg_f, video_f
            return output, logit_eeg, logit_video

        elif self.args.coral_loss:
            return output, eeg_f, video_f

        else:
            return output
