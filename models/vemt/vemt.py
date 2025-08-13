import torch
import torch.nn as nn
import torch.nn.functional as F
from .modeling_finetune import VisionTransformer
from .mamba_models import AudioMamba


class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCNLayer, self).__init__()

        ## TODO: Test simple and deeper layer after GCN calculation
        self.layers = nn.Sequential(nn.Linear(input_dim, output_dim), nn.LeakyReLU(0.2))
        # self.layers = nn.Sequential(
        #     nn.Linear(input_dim, input_dim),
        #     nn.GELU(),
        #     nn.LayerNorm(input_dim),
        #     nn.Linear(input_dim, output_dim)
        # )
    
        self.layers.apply(self.init_weights_classify)
    
    def init_weights_classify(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x, adj):
        x = torch.matmul(adj, x)
        x = self.layers(x)
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


def build_normalized_adj(x, adj_mask, eps=1e-3, self_loop=1.0, keep_neg=False):
    """
    Compute a symmetric normalized adjacency matrix from node features and a mask.

    Args:
        x: Tensor, shape [B, N, D] - Node features
        adj_mask: Tensor, shape [N, N] - Binary mask for allowed connections
        eps: float - Minimum degree clamp to avoid division by zero
        self_loop: float - Value to add to self-connections (I)
        keep_neg: bool - If False, negative similarities are clamped to zero

    Returns:
        A: Tensor, shape [B, N, N] - Symmetric normalized adjacency
    """
    # L2-normalize
    x_norm = F.normalize(x, p=2, dim=-1)  # [B, N, D]
    # Similarity matrix
    S = torch.matmul(x_norm, x_norm.transpose(1, 2))  # [B, N, N]

    if not keep_neg:
        S = S.clamp_min(0.0)

    S = torch.where(adj_mask.bool().unsqueeze(0), S, torch.zeros_like(S))

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


class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x, adj):
        x = torch.matmul(adj, x)  # Graph convolution: aggregate neighbors
        return self.layers(x)


class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, type="local"):
        super().__init__()
        self.type = type
        self.gcn1 = GCNLayer(input_dim, input_dim)
        self.gcn2 = GCNLayer(input_dim, output_dim)

        if self.type == "region":
            self.weight = nn.Parameter(torch.tensor([[0.5, 0.5]]))

    def compute_local_adj(self, B, C, region_indices, device):
        """Build adjacency mask for local EEG channels."""
        adj_mask = torch.zeros(C, C, device=device)
        for ch_group in region_indices:
            for i in ch_group:
                for j in ch_group:
                    if i != j:
                        adj_mask[i, j] = 1.0
        return adj_mask

    def compute_region_adj(self, N, device):
        """Build adjacency mask for region graph with globals and local regions."""
        adj_mask = torch.zeros(N, N, device=device)
        # global-to-global
        adj_mask[0, 1] = adj_mask[1, 0] = 1
        # globals <-> regions
        adj_mask[0, 2:] = adj_mask[1, 2:] = 1
        adj_mask[2:, 0] = adj_mask[2:, 1] = 1
        # region <-> region
        if N > 2:
            adj_mask[2:, 2:] = 1
            ridx = torch.arange(2, N, device=device)
            adj_mask[ridx, ridx] = 0
        return adj_mask

    def forward(self, x, region_indices=None):
        if self.type == "local":
            B, C, D = x.shape
            adj_mask = self.compute_local_adj(B, C, region_indices, x.device)
            adj = build_normalized_adj(x, adj_mask, eps=1e-3, self_loop=0.0, keep_neg=True)
            x1 = self.gcn1(x, adj)
            x1 = 0.5 * x1 + 0.5 * x  # residual connection
            x1 = self.gcn2(x1, adj)
            return x1  # [B, C, D_out]

        elif self.type == "region":
            B, N, D = x.shape
            adj_mask = self.compute_region_adj(N, x.device)
            adj = build_normalized_adj(x, adj_mask, eps=1e-3, self_loop=0.0, keep_neg=True)
            x1 = self.gcn1(x, adj)
            x1 = 0.5 * x1 + 0.5 * x
            x1 = self.gcn2(x1, adj)

            # Fuse global video and EEG nodes
            global_f1 = x1[:, 0, :]  # video global
            global_f2 = x1[:, 1, :]  # eeg global
            stacked = torch.stack([global_f1, global_f2], dim=1)  # [B, 2, D_out]
            out = torch.bmm(self.weight.expand(B, -1, -1), stacked).squeeze(1)
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

        if output_dim[1] == 1:
            self.output_dim = (output_dim[0],)

        if self.args.set_video_only or self.args.eeg_signal: # VideoMAEv2 (https://github.com/OpenGVLab/VideoMAEv2/blob/master/models/modeling_finetune.py)
            self.video_model = VisionTransformer(args=args,
                img_size=image_size,
                patch_size=16,
                in_chans=3,
                output_dim= output_dim,
                all_frames=32,
                embed_dim=self.embed_dim,
                depth=12,
                num_heads=12,
                tubelet_size=2,
            )

        if self.args.set_eeg_only or self.args.eeg_signal: # Audio-Mamba (https://github.com/kaistmm/Audio-Mamba-AuM/blob/b0eb85d595c4f079606dfff4f4c6f709caafa858/src/models/mamba_models.py)
            self.eeg_model = AudioMamba(args=args, spectrogram_size=self.spectrogram_size, depth=24, channels=self.in_channel, output_dim=output_dim)

        # E+V Murged classifier Head
        if self.args.eeg_signal:
            if self.args.gcn:
                self.eeg_feat_proj = nn.Linear(self.spectrogram_size[0] * self.spectrogram_size[1], self.embed_dim)

                self.gcn_local = GCN(input_dim=self.embed_dim, output_dim=self.embed_dim, type= "local")
                self.gcn_region = GCN(input_dim=self.embed_dim, output_dim=self.num_classes, type = "region")
                self.region_pool = RegionalAttentionPool(d_model=self.embed_dim)
            
            else:
                self.classifier = nn.Linear(self.hs, self.num_classes)
                self.classifier.apply(self.init_weights_classify)


    def init_weights_classify(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    
    def forward(self, x):
        eeg = x["eeg"]
        eeg_raw = eeg
        video = x["video"].transpose_(-3, -4)
        
        if self.args.set_eeg_only:
            output = self.eeg_model(eeg)

        elif self.args.set_video_only:
            output = self.video_model(video)

        elif self.args.eeg_signal:
            eeg_f = self.eeg_model(eeg)
            video_f = self.video_model(video)

            if self.args.gcn:
                B, ch, Freq, Time = eeg_raw.shape
                eeg_flat = eeg_raw.view(B * ch, Freq * Time)
                eeg_local = self.eeg_feat_proj(eeg_flat).view(B, ch, -1)  # [B, ch, D]

                if self.args.dataset == "eav":
                    region_dataset = [
                    [0,1,2,3,4,5,6,7,8,9],       # Frontal
                    [14,18],                     # Temporal
                    [10,11,12,13,15,16,17],      # Central
                    [19,20,21,22,23,24,25,26],   # Parietal
                    [27,28,29,30],]              # Occipital
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

                eeg_gcn = self.gcn_local(eeg_local, region_indices = region_dataset)
                eeg_region = self.region_pool(eeg_gcn, region_indices = region_dataset)

                # Concat all nodes
                all_nodes = torch.cat([
                    video_f.unsqueeze(1),
                    eeg_f.unsqueeze(1),
                    eeg_region
                ], dim=1)  # [B, ch+2, D]

                output = self.gcn_region(all_nodes)

            else:
                fused_f = torch.cat((eeg_f, video_f), dim=1)
                output = self.classifier(fused_f)
                
            if self.args.dataset == 'emognition' or 'mdmer':
                output_v = output[:, :self.output_dim[0]].unsqueeze(-1)
                output_a = output[:, self.output_dim[0]:].unsqueeze(-1)
                output = torch.concat((output_v, output_a), dim=-1)

        return output