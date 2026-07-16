import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .CBraMod import CBraMod_Model


# --vemt_video string -> (module, wrapper class) implementing the AdaMAE/VideoMAE
# VisionTransformer interface.
_VIDEO_BACKBONE_REGISTRY = {
    "ViViT":    ("modeling_vivit",      "VivitVisionTransformer"),
    "TSF":      ("modeling_tsf",        "TSFVisionTransformer"),
    "Swin":     ("modeling_swin",       "SwinVisionTransformer"),
    "AdaMAE":   ("modeling_finetune_v0", "VisionTransformer"),
    "VideoMAE": ("modeling_finetune",   "VisionTransformer"),
}


def _build_video_backbone(args, image_size, output_dim, embed_dim):
    """Instantiate the video backbone selected by args.vemt_video."""
    import importlib
    name = getattr(args, "vemt_video", "VideoMAE")
    if name not in _VIDEO_BACKBONE_REGISTRY:
        name = "VideoMAE"
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
                         temporal_weight=None, angular=False, edge_scale=None):
    """Symmetric normalized adjacency A = D^-1/2 S D^-1/2 from node features + mask.

    temporal_weight: optional [N, N] multiplied into the similarity before
    masking (Gaussian temporal-distance prior; 1.0 leaves an edge unaffected).
    angular: use MMGCN angular similarity 1 - arccos(cos)/pi in [0,1] instead of
    raw cosine (more discriminative, always non-negative).
    edge_scale: optional [N, N] multiplied into the similarity after masking
    (e.g. a learnable per-relation scale like MMGCN's cross-modal gamma).
    """
    x_norm = F.normalize(x, p=2, dim=-1)  # [B, N, D]
    S = torch.matmul(x_norm, x_norm.transpose(1, 2))  # [B, N, N]

    if angular:
        S = 1.0 - torch.arccos(S.clamp(-1.0 + 1e-6, 1.0 - 1e-6)) / math.pi
    elif not keep_neg:
        S = S.clamp_min(0.0)

    S = torch.where(adj_mask.bool().unsqueeze(0), S, torch.zeros_like(S))

    if edge_scale is not None:
        S = S * edge_scale.to(S.device).unsqueeze(0)

    if temporal_weight is not None:
        S = S * temporal_weight.to(S.device).unsqueeze(0)

    S = 0.5 * (S + S.transpose(1, 2))

    if self_loop and self_loop > 0:
        I = torch.eye(S.size(1), device=S.device).unsqueeze(0)
        S = S + self_loop * I

    deg = S.sum(dim=-1).clamp_min(eps)
    D_inv_sqrt = deg.pow(-0.5)
    D_inv_sqrt = torch.diag_embed(D_inv_sqrt)
    A = D_inv_sqrt @ S @ D_inv_sqrt

    A = torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

    return A


class GCN(nn.Module):
    def __init__(self, args, input_dim, output_dim, type="local"):
        super().__init__()
        self.type = type
        self.args = args

        self.gcn1 = GCNLayer(input_dim, input_dim)
        self.gcn2 = GCNLayer(input_dim, output_dim)
        self.gcn1_norm = nn.LayerNorm(input_dim)

        if self.type == "region":
            # --fusion_gate_fixed <p> pins the video-share to a CONSTANT p and freezes
            # it (requires_grad=False) → out=(1-p)*eeg+p*video with p fixed, so the
            # video branch always gets p of the gradient (counters modality imbalance).
            # Default (None) keeps the legacy learnable -1.0 logit (=0.269).
            _gf = getattr(args, 'fusion_gate_fixed', None)
            if _gf is not None:
                self.weight = nn.Parameter(
                    torch.full((output_dim,), math.log(_gf / (1.0 - _gf))), requires_grad=False)
            else:
                self.weight = nn.Parameter(torch.full((output_dim,), -1.0))

            # --fusion_gate_adaptive: per-sample video-share bounded around 0.5,
            # w = 0.5 + beta*tanh(mlp([video_logit, eeg_logit])). Bounding keeps it
            # near 0.5 so it cannot collapse to the EEG-dominant corner.
            if getattr(args, 'fusion_gate_adaptive', False):
                self.gate_beta = float(getattr(args, 'fusion_gate_beta', 0.3))
                self.gate_mlp = nn.Sequential(
                    nn.Linear(output_dim * 2, output_dim), nn.GELU(),
                    nn.Linear(output_dim, output_dim))
                nn.init.zeros_(self.gate_mlp[-1].weight)
                nn.init.zeros_(self.gate_mlp[-1].bias)

            # Per-sample gate shift from the video-eeg agreement cos(video, eeg).
            if getattr(args, 've_gate', False):
                self.ve_gate = nn.Sequential(
                    nn.Linear(1, 16), nn.GELU(), nn.Linear(16, output_dim))
                nn.init.zeros_(self.ve_gate[-1].weight)
                nn.init.zeros_(self.ve_gate[-1].bias)

            # Shared projection for the cross-modal adjacency (identity-init).
            if getattr(args, 'd2_align', False):
                self.align_proj = nn.Linear(input_dim, input_dim)
                nn.init.eye_(self.align_proj.weight)
                nn.init.zeros_(self.align_proj.bias)

            # Learnable Gaussian std (log space) for --gcn_temporal_adj prior.
            self.log_sigma = nn.Parameter(torch.log(torch.tensor(3.0)))

            # --gcn_ii: GCNII propagation (Chen et al. ICML2020, as used by MMGCN)
            # for the joint video/eeg graph. Initial residual to H^(0) keeps node
            # identity in a BOUNDED convex combination (unlike the self-loop, which
            # compounds and explodes), and identity mapping lets it go deep without
            # oversmoothing so cross-modal edges actually propagate. gamma is a
            # learnable scale on cross-modal edges (MMGCN); init 1.0 = no-op start.
            if getattr(args, 'gcn_ii', False):
                h = int(getattr(args, 'gcn_ii_hidden', 256))
                L = int(getattr(args, 'gcn_ii_layers', 4))
                self.ii_alpha = float(getattr(args, 'gcn_ii_alpha', 0.1))
                self.ii_eta = float(getattr(args, 'gcn_ii_eta', 0.5))
                self.ii_in = nn.Linear(input_dim, h)
                self.ii_w = nn.ModuleList([nn.Linear(h, h, bias=False) for _ in range(L)])
                self.ii_head = nn.Linear(h, output_dim)
                self.ii_gamma = nn.Parameter(torch.tensor(1.0))

        # video-clip GCN (--gcn_video_local) uses the same temporal prior.
        if self.type == "local":
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

        Node layout: [0:K_v) video clips, [K_v:K_v+K_e) EEG clips,
        [K_v+K_e:n_main) regions, [n_main:N) optional prototypes (legacy).
        Cross-modal edges pair video[i]<->EEG[i] when K_v==K_e, else all-to-all.
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

        # all clip globals <-> regions; region <-> region.
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
        t_diff_sq = (t.unsqueeze(0) - t.unsqueeze(1)) ** 2     # [N, N]

        sigma = self.log_sigma.exp().clamp(min=0.5)
        gauss = torch.exp(-t_diff_sq / (2.0 * sigma * sigma))
        # Non-temporal pairs (region-anything): weight 1.0 → no modulation.
        return torch.where(both_time, gauss, torch.ones_like(gauss))

    def forward(self, x, region_indices=None, proto_dim_sizes=(),
                num_video_nodes=1, num_eeg_nodes=1, node_time_pos=None):
        if self.type == "local":
            B, C, D = x.shape
            adj_mask = self.compute_local_adj(B, C, region_indices, x.device)
            temporal_weight = None
            if node_time_pos is not None and getattr(self.args, 'gcn_temporal_adj', False):
                temporal_weight = self._compute_temporal_weight(node_time_pos, x.device)
            adj = build_normalized_adj(x, adj_mask, eps=1e-3, self_loop=0.0, keep_neg=True,
                                       temporal_weight=temporal_weight)
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

            # With a self-loop, gcn2 keeps each node's own feature instead of
            # averaging it over neighbors; the raw residual then carries large
            # backbone magnitudes straight into the logits and blows them up on
            # big clip graphs. Normalize node scale first (param-free, cosine adj
            # unchanged) so self-loop is stable. Off by default -> base untouched.
            if getattr(self.args, 'gcn_self_loop', 0.0) > 0:
                x = F.layer_norm(x, (D,))

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

            if getattr(self.args, 'gcn_ii', False):
                # GCNII on the joint graph: angular-similarity adjacency with a
                # learnable cross-modal scale gamma, self-loop, temporal prior.
                x_ln = F.layer_norm(x, (D,))
                cross = torch.zeros(N, N, device=x.device)
                cross[:K_v, K_v:] = 1.0          # video -> eeg/region
                cross[K_v:, :K_v] = 1.0          # eeg/region -> video
                edge_scale = 1.0 + (self.ii_gamma - 1.0) * cross
                adj = build_normalized_adj(
                    x_ln, adj_mask, eps=1e-3, self_loop=1.0, angular=True,
                    temporal_weight=temporal_weight, edge_scale=edge_scale,
                )
                H0 = self.ii_in(x_ln)
                H = H0
                for l, wl in enumerate(self.ii_w):
                    beta = math.log(self.ii_eta / (l + 1) + 1.0)
                    PH = torch.bmm(adj, H)
                    tmp = (1.0 - self.ii_alpha) * PH + self.ii_alpha * H0
                    H = (1.0 - beta) * tmp + beta * wl(tmp)   # (1-b)I + bW
                    H = F.gelu(H)
                if getattr(self.args, 'dump_affinity', False):
                    self._dump_adj = adj.detach()      # [B,N,N] angular+gamma adjacency
                    self._dump_H = H.detach()          # [B,N,hidden] final propagated nodes
                g1 = H[:, :K_v, :].mean(dim=1) if K_v > 1 else H[:, 0, :]
                g2 = H[:, K_v:K_v + K_e, :].mean(dim=1) if K_e > 1 else H[:, K_v, :]
                # Decorrelation: the dump showed video/eeg node features collapse
                # (cos~0.92) after propagation, leaving nothing to fuse. Penalize
                # their similarity so the two streams stay distinct. g1/g2 are on
                # the main path (no DDP unreachable-param issue).
                _dw = float(getattr(self.args, 'gcn_ii_decorr', 0.0))
                if _dw > 0 and self.training:
                    self._fusion_aux = _dw * F.cosine_similarity(g1, g2, dim=-1).pow(2).mean()
                global_f1 = self.ii_head(g1)
                global_f2 = self.ii_head(g2)
            else:
                _d2 = getattr(self.args, 'd2_align', False)
                adj = build_normalized_adj(
                    self.align_proj(x) if _d2 else x, adj_mask,
                    eps=1e-3, self_loop=getattr(self.args, 'gcn_self_loop', 0.0),
                    keep_neg=True, temporal_weight=temporal_weight,
                )
                if _d2 and self.training:
                    _vp = F.normalize(self.align_proj(x[:, :K_v, :].mean(dim=1)), dim=-1)
                    _ep = F.normalize(self.align_proj(x[:, K_v, :]), dim=-1)
                    _tau = float(getattr(self.args, 'd2_align_tau', 0.1))
                    import torch.distributed as _d
                    if (getattr(self.args, 'd2_gather', False) and _d.is_available()
                            and _d.is_initialized() and _d.get_world_size() > 1):
                        # gather negatives across GPUs, keep the local slot grad-connected
                        def _gg(t):
                            g = [torch.zeros_like(t) for _ in range(_d.get_world_size())]
                            _d.all_gather(g, t.contiguous()); g[_d.get_rank()] = t
                            return torch.cat(g, 0)
                        _vp_all, _ep_all = _gg(_vp), _gg(_ep)
                        _off = _d.get_rank() * _vp.size(0)
                    else:
                        _vp_all, _ep_all, _off = _vp, _ep, 0
                    _lg1 = _vp @ _ep_all.t() / _tau
                    _lg2 = _ep @ _vp_all.t() / _tau
                    _lab = torch.arange(_vp.size(0), device=_vp.device) + _off
                    _nce = 0.5 * (F.cross_entropy(_lg1, _lab) + F.cross_entropy(_lg2, _lab))
                    self._fusion_aux = float(getattr(self.args, 'd2_align_w', 0.3)) * _nce
                # Node-feature dropout into each GCN layer (adjacency stays clean).
                _dp = float(getattr(self.args, 'gcn_dropout', 0.0))
                h = F.dropout(x, _dp, self.training) if _dp > 0 else x
                x1 = self.gcn1(h, adj)
                x1 = F.gelu(x1)
                x1 = self.gcn1_norm(x1)
                x1 = 0.5 * x1 + 0.5 * x
                if getattr(self.args, 'dump_affinity', False):
                    self._dump_adj = adj.detach()      # [B,N,N]
                    self._dump_H = x1.detach()         # post-gcn1 768-d node features
                    self._dump_Hpre = x.detach()       # pre-GCN node features
                if _dp > 0:
                    x1 = F.dropout(x1, _dp, self.training)
                x1 = self.gcn2(x1, adj)

                # Pool post-GCN nodes per modality: video=global_f1, eeg=global_f2.
                if K_v > 1:
                    global_f1 = x1[:, :K_v, :].mean(dim=1)
                else:
                    global_f1 = x1[:, 0, :]
                if K_e > 1:
                    global_f2 = x1[:, K_v:K_v + K_e, :].mean(dim=1)
                else:
                    global_f2 = x1[:, K_v, :]

                if getattr(self.args, 'video_aux_warmup', 0) > 0:
                    self._video_logit = global_f1

            if getattr(self.args, 've_gate', False):
                agree = F.cosine_similarity(global_f1, global_f2, dim=-1).unsqueeze(-1)
                w = torch.sigmoid(self.weight + self.ve_gate(agree))
            elif getattr(self.args, 'fusion_gate_adaptive', False):
                w = 0.5 + self.gate_beta * torch.tanh(
                    self.gate_mlp(torch.cat([global_f1, global_f2], dim=-1)))
            else:
                w = torch.sigmoid(self.weight)                    # [D_out] (static)
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
            # EEG backbone (--eeg_backbone): 'cbramod' (default), 'reve', or 'labram'.
            _eeg_backbone = getattr(args, 'eeg_backbone', 'cbramod')
            if _eeg_backbone == 'reve':
                from .REVE import REVE_Model
                self.eeg_model = REVE_Model(args, output_dim=output_dim, in_chans=self.in_channel)
            elif _eeg_backbone == 'labram':
                from .LaBraM_VEMT import LaBraM_VEMT_Model
                self.eeg_model = LaBraM_VEMT_Model(args, output_dim=output_dim, in_chans=self.in_channel)
            else:
                self.eeg_model = CBraMod_Model(args, output_dim=output_dim, in_chans=self.in_channel)

        if self.args.eeg_signal:
            if self.args.gcn:
                # Per-channel EEG feature source for GCN region nodes
                # (--gcn_region_eeg_source): 'stft' (STFT → Linear) or 'cbramod'
                # (pre-positional patch_embedding output), projected 'linear' or 'conv'.
                _eeg_src = getattr(args, 'gcn_region_eeg_source', 'stft')
                _eeg_proj_kind = getattr(args, 'gcn_region_eeg_proj', 'linear')
                # Backbone per-channel dim: CBraMod 200, REVE-base 512.
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
                # --deep_fuse: stage-1 per-modality classifier heads (video clip
                # branch + EEG region branch). The joint GCN keeps its own logit;
                # final = 0.25*video + 0.25*eeg + 0.5*fused with deep supervision.
                if getattr(args, 'deep_fuse', False):
                    self.df_video_head = nn.Linear(self.embed_dim, self.num_classes)
                    self.df_eeg_head = nn.Linear(self.embed_dim, self.num_classes)
                    self.deep_fuse_w = float(getattr(args, 'deep_fuse_w', 0.5))
                    if getattr(args, 'deep_fuse_learn_w', False):
                        init_w = torch.log(torch.tensor([0.25, 0.25, 0.5]))
                        self.df_logit_w = nn.Parameter(init_w.view(3, 1).repeat(1, self.num_classes))
                    if getattr(args, 'deep_fuse_adapt_w', False):
                        self.df_adapt = nn.Sequential(
                            nn.Linear(3 * self.num_classes, 64), nn.GELU(), nn.Linear(64, 3))
                # --gcn_video_local: refine video clip nodes among themselves before
                # the joint graph (mirrors gcn_local for EEG channels).
                if getattr(args, 'gcn_video_local', False):
                    self.gcn_video_local = GCN(args, input_dim=self.embed_dim, output_dim=self.embed_dim, type="local")
                    if getattr(args, 'gcn_video_local_attn', False):
                        self.gcn_video_local_pool = nn.Linear(self.embed_dim, 1)

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

        self.eeg_proj = nn.Linear(200, 768)

        if self.args.fusion == 'FiLM':
            self.film = FiLM(dim=self.embed_dim)
        elif self.args.fusion == 'attention':
            self.cross_attn = CrossAttentionFusion(self.embed_dim)

        # Clip attention pooling: per-modality scalar score per clip → softmax weighted sum.
        _multi_video = (getattr(args, 'num_clips', 1) > 1
                        or getattr(args, 'dense_video_clips', False))
        _multi_eeg = (getattr(args, 'num_clips', 1) > 1)
        if getattr(args, 'clip_pool', 'mean') == 'attn':
            if (self.args.set_video_only or self.args.eeg_signal) and _multi_video:
                self.clip_attn_v = nn.Linear(self.embed_dim, 1)
                nn.init.normal_(self.clip_attn_v.weight, std=0.5)
                nn.init.zeros_(self.clip_attn_v.bias)
            if (self.args.set_eeg_only or self.args.eeg_signal) and _multi_eeg:
                self.clip_attn_e = nn.Linear(self.embed_dim, 1)
                nn.init.normal_(self.clip_attn_e.weight, std=0.5)
                nn.init.zeros_(self.clip_attn_e.bias)

        # Dense video feature cache (block-split intermediate): in-memory per-rank
        # dict + optional on-disk tier shared across ranks/epochs (setup_dense_cache).
        # Keyed by dataset idx; values stored as fp16 CPU [N_actual, P, D].
        self._video_block_cache = {}
        self._video_block_cache_dir = None

    def freeze_backbones(self, video_unfreeze_last_n: int = 0, eeg_unfreeze_last_n: int = 0, eeg_full_unfreeze: bool = False):
        """Freeze pretrained backbones, keep classifier / fusion heads trainable.

        video_unfreeze_last_n / eeg_unfreeze_last_n: keep last N blocks (+ norm/
        head, resp. classifier) trainable. eeg_full_unfreeze overrides the EEG
        side and unfreezes all of CBraMod.
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
            "classifier",  # VEMT classifier when gcn=False
            "film",
            "cross_attn",
            "df_video_head",   # --deep_fuse stage-1 branch heads
            "df_eeg_head",
            "df_adapt",
            "gcn_video_local",
        ]

        for module_name in head_module_names:
            if hasattr(self, module_name):
                module = getattr(self, module_name)
                for p in module.parameters():
                    p.requires_grad = True

        if hasattr(self, "df_logit_w"):
            self.df_logit_w.requires_grad = True

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


    def _va_to_flat(self, logit):
        if logit.dim() == 3 and logit.size(-1) == 2:
            return torch.cat([logit[:, :, 0], logit[:, :, 1]], dim=1)
        return logit

    def _flat_to_va(self, logit):
        if logit.dim() == 2 and self.args.dataset in ('emognition', 'mdmer'):
            n = self.output_dim[0]
            return torch.stack([logit[:, :n], logit[:, n:]], dim=-1)
        return logit

    def _pool_clips(self, feat_flat, logit_flat, B, K, modality='video'):
        """Pool per-clip features and logits across the K dimension.

        Returns (logit, feat) with K removed. Pool method: self.args.clip_pool.
        """
        pool = getattr(self.args, 'clip_pool', 'mean')
        feats  = feat_flat.view(B, K, *feat_flat.shape[1:])
        logits = logit_flat.view(B, K, *logit_flat.shape[1:])

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
            and n_trainable >= 0
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

        pool_mode = getattr(self.args, 'clip_pool', 'mean')

        if pool_mode == 'attn' and hasattr(self, 'clip_attn_v'):
            scores = self.clip_attn_v(feat).squeeze(-1)  # [B, N_max]
            scores = scores.masked_fill(mask == 0, float('-inf'))
            w = torch.softmax(scores, dim=1)              # [B, N_max]
            # All-padded row → softmax(-inf) = NaN; fall back to zeros.
            w = torch.nan_to_num(w, nan=0.0)
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
                          f'score_std={_s_std:.4e}, '
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

    def _run_eeg_backbone(self, x):
        """Run EEG backbone; K-clip 5D input is flattened, pooled back across K.

        CBraMod input is 4D at K=1; K-clip bumps "eeg"/"eeg_local" to 5D
        ([B, K, Ch, T_seg, seg_len]). Returns (logit, feat).
        """
        _multi = (x["eeg"].dim() == 5)

        if not _multi:
            return self.eeg_model(x)

        B, K = x["eeg"].shape[:2]
        x_flat = dict(x)
        for key in ("eeg", "eeg_local"):
            if key in x and torch.is_tensor(x[key]):
                v = x[key]
                x_flat[key] = v.view(B * K, *v.shape[2:])

        logit_flat, feat_flat = self.eeg_model(x_flat)
        logit, feat = self._pool_clips(feat_flat, logit_flat, B, K, modality='eeg')
        return logit, feat

    def forward(self, x):
        eeg = x["eeg"]
        video = x["video"].transpose_(-3, -4)
        eeg_stft = x["eeg_local"]

        # Dense N-clip video mode: 'video_lengths' (added by dense_video_collate_fn)
        # is the activation signal → pooled video global.
        video_lengths = x.get("video_lengths", None)
        sample_idxs = x.get("sample_idx", None)  # for cross-epoch frozen-feature cache
        _dense_active = (video_lengths is not None)
        _dense_chunk = int(getattr(self.args, 'dense_chunk_size', 6))

        if self.args.set_eeg_only:
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
                logit_flat, feat_flat = self.video_model(flat, return_feat=True)
                output_pool, _ = self._pool_clips(feat_flat, logit_flat, B, K, modality='video')
                # video_model keeps logits flat when num_clips>1; apply V/A split here.
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
                # Head may leave logits flat [B, n_v+n_a]; reshape to [B, n_v, 2].
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
            # --gcn_video_per_clip: expose each video clip as its own GCN node
            # (needs a multi-clip video dim: 6D K-clip or dense N-clip).
            _use_per_clip = (
                getattr(self.args, 'gcn_video_per_clip', False)
                and getattr(self.args, 'gcn', False)
                and (video.dim() == 6 or _dense_active)
            )

            video_clips = None  # [B, K_v, D] when per-clip video is active

            eeg_logit, eeg_f = self._run_eeg_backbone(x)

            if _dense_active:
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

            if self.args.fusion == 'attention':
                video_f = self.cross_attn(eeg_f, video_f)

            elif self.args.fusion == 'FiLM':
                video_f = self.film(video_f, eeg_f)

            fused_f = torch.cat((eeg_f, video_f), dim=1)

            if self.args.gcn:
                if getattr(self.args, 'gcn_region_eeg_source', 'stft') == 'cbramod':
                    # Per-channel features for GCN region nodes.
                    # Shape contract: [B, ch, T_seg=10, 200] regardless of backbone.
                    # CBraMod uses pre-positional patch_embedding output; REVE uses
                    # return_per_channel_pre (post-encoder) — same shape downstream.
                    pe_input = x["eeg"]
                    _is_reve = getattr(self.args, 'eeg_backbone', 'cbramod') == 'reve'
                    if _is_reve:
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

                eeg_gcn = self.gcn_local(eeg_local, region_indices=region_dataset)

                eeg_region = self.region_pool(eeg_gcn, region_indices = region_dataset)

                if self.use_region_importance:
                    eeg_region, region_alpha = self.apply_region_importance(eeg_region, video_f, eeg_f)

                video_node = video_f
                eeg_node = eeg_f

                # Video node(s): K clip nodes when per-clip active, else single node.
                if _use_per_clip and video_clips is not None:
                    v_clip_nodes = video_clips
                else:
                    v_clip_nodes = video_node.unsqueeze(1)  # [B, 1, D]
                num_video_nodes = v_clip_nodes.size(1)

                # --gcn_video_local (stage 1b): refine clips with a Gaussian-temporal
                # clip GCN and pool to a single video node, so the joint graph holds
                # [1 video, 1 eeg, R regions] instead of K_v flooding video clips.
                if getattr(self.args, 'gcn_video_local', False) and num_video_nodes > 1:
                    _vt = torch.arange(num_video_nodes, device=v_clip_nodes.device)
                    if getattr(self.args, 'gcn_clip_pe', False):
                        v_clip_nodes = v_clip_nodes + self._sinusoid_pe(
                            num_video_nodes, v_clip_nodes.size(-1), v_clip_nodes.device).unsqueeze(0)
                    v_clip_nodes = self.gcn_video_local(v_clip_nodes, region_indices=None, node_time_pos=_vt)
                    if getattr(self.args, 'gcn_video_local_attn', False):
                        _a = torch.softmax(self.gcn_video_local_pool(v_clip_nodes), dim=1)  # [B, Kv, 1]
                        v_clip_nodes = (_a * v_clip_nodes).sum(dim=1, keepdim=True)          # [B, 1, D]
                    else:
                        v_clip_nodes = v_clip_nodes.mean(dim=1, keepdim=True)                # [B, 1, D]
                    num_video_nodes = 1

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

                if getattr(self.args, 'deep_fuse', False):
                    # Stage-1 branch logits (before the joint graph collapses the
                    # modalities): pool the video clip nodes and the EEG region
                    # nodes to one vector each, classify separately, then combine
                    # with the joint (fused) logit 0.25/0.25/0.5.
                    fused_logit = output
                    video_logit = self.df_video_head(v_clip_nodes.mean(dim=1))
                    eeg_logit = self.df_eeg_head(eeg_region.mean(dim=1))
                    if getattr(self.args, 'deep_fuse_adapt_w', False):
                        w = torch.softmax(
                            self.df_adapt(torch.cat([video_logit, eeg_logit, fused_logit], dim=-1)), dim=-1)
                        output = (w[:, 0:1] * video_logit + w[:, 1:2] * eeg_logit
                                  + w[:, 2:3] * fused_logit)
                    elif getattr(self.args, 'deep_fuse_learn_w', False):
                        w = torch.softmax(self.df_logit_w, dim=0)
                        output = w[0] * video_logit + w[1] * eeg_logit + w[2] * fused_logit
                    else:
                        output = 0.25 * video_logit + 0.25 * eeg_logit + 0.5 * fused_logit
                    self._deep_fuse_branches = (video_logit, eeg_logit, fused_logit)

            else:
                fused_f = F.normalize(fused_f, dim=-1)
                output = self.classifier(fused_f)

            if self.args.dataset in ('emognition', 'mdmer'):

                # output: [B, 2C] -> [B, C, 2]
                if output.dim() == 2:
                    output_v = output[:, :self.output_dim[0]].unsqueeze(-1)
                    output_a = output[:, self.output_dim[0]:].unsqueeze(-1)
                    output = torch.cat((output_v, output_a), dim=-1)

        return output
