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


class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, type = "local"):
        super(GCN, self).__init__()
        self.type = type

        self.gcn1 = GCNLayer(input_dim, input_dim)
        self.gcn2 = GCNLayer(input_dim, output_dim)

        self.attn_fc = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 1)
        )
        if self.type == "region":
            self.weight = nn.Parameter(torch.tensor([[0.5, 0.5]]))  # [1, 2]

    def compute_attention_local(self, x, region_indices):
        B, C, D = x.shape

        # pairwise attention
        x_i = x.unsqueeze(2).expand(B, C, C, D)
        x_j = x.unsqueeze(1).expand(B, C, C, D)
        pairwise = torch.cat([x_i, x_j], dim=-1)     # [B, C, C, 2D]
        attn_scores = self.attn_fc(pairwise).squeeze(-1)  # [B, C, C]

        adj_mask = torch.zeros(C, C, device=x.device)
        for ch_idxs in region_indices:
            for i in ch_idxs:
                for j in ch_idxs:
                    if i != j:
                        adj_mask[i, j] = 1.0

        idx = torch.arange(C, device=x.device)
        adj_mask[idx, idx] = 0

        # Apply mask (block disallowed connections)
        attn_scores = attn_scores.masked_fill(adj_mask[None, :, :] == 0, float('-inf'))

        adj = F.softmax(attn_scores, dim=-1)  # [B, N, N]

        return x, adj

    def compute_attention_region(self, x):
        # x: [B, N, D]
        B, N, D = x.shape

        x_i = x.unsqueeze(2).expand(B, N, N, D)
        x_j = x.unsqueeze(1).expand(B, N, N, D)
        pairwise = torch.cat([x_i, x_j], dim=-1)
        attn_scores = self.attn_fc(pairwise).squeeze(-1)  # [B, N, N]

        adj_mask = torch.zeros(N, N, device=x.device)
        # globals
        adj_mask[0, 1] = 1; adj_mask[1, 0] = 1
        # globals <-> regions
        adj_mask[0, 2:] = 1; adj_mask[1, 2:] = 1
        adj_mask[2:, 0] = 1; adj_mask[2:, 1] = 1
        # region <-> region
        if N > 2:
            adj_mask[2:, 2:] = 1
            ridx = torch.arange(2, N, device=x.device)
            adj_mask[ridx, ridx] = 0

        attn_scores = attn_scores.masked_fill(adj_mask[None, :, :] == 0, float('-inf'))
        all_neg_inf = torch.isinf(attn_scores).all(dim=-1, keepdim=True)
        attn_scores = torch.where(all_neg_inf, torch.zeros_like(attn_scores), attn_scores)
        adj = F.softmax(attn_scores, dim=-1)
        return x, adj

    def forward(self, x, region_indices=None):
        if self.type == "local":
            x, adj = self.compute_attention_local(x, region_indices)  # [B,C,D], [B,C,C]
            x1 = self.gcn1(x, adj)
            x1 = 0.5 * x1 + 0.5 * x
            x1 = self.gcn2(x1, adj)
            return x1  # [B, C, D_out]

        elif self.type == "region":
            # x: [B, 2+R, D]
            x, adj = self.compute_attention_region(x)   # [B,N,D], [B,N,N]
            x1 = self.gcn1(x, adj)
            x1 = 0.5 * x1 + 0.5 * x
            x1 = self.gcn2(x1, adj)

            # global fusion
            global_f1 = x1[:, 0, :]
            global_f2 = x1[:, 1, :]
            stacked = torch.stack([global_f1, global_f2], dim=1)      # [B,2,D_out]
            out = torch.bmm(self.weight.expand(x.size(0), -1, -1), stacked).squeeze(1)
            return out  # [B, D_out]
        else:
            raise ValueError(self.type)

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
        # self.num_classes = output_dim[0]
        self.in_channel = eeg_channels

        if output_dim[1] == 1:
            self.output_dim = (output_dim[0],)

        if self.args.set_video_only or self.args.eeg_signal: # VideoMAEv2 (https://github.com/OpenGVLab/VideoMAEv2/blob/master/models/modeling_finetune.py)
            self.video_model = VisionTransformer(args=args,
                img_size=image_size,
                patch_size=16,
                in_chans=3,
                num_classes= self.num_classes,
                all_frames=32,
                embed_dim=self.embed_dim,
                depth=12,
                num_heads=12,
                tubelet_size=2,
            )

        if self.args.set_eeg_only or self.args.eeg_signal: # Audio-Mamba (https://github.com/kaistmm/Audio-Mamba-AuM/blob/b0eb85d595c4f079606dfff4f4c6f709caafa858/src/models/mamba_models.py)
            self.eeg_model = AudioMamba(args=args, spectrogram_size=self.spectrogram_size, depth=24, channels=self.in_channel, num_classes=self.num_classes)
            # self.eeg2fps = nn.Conv2d(eeg_channels * 2, eeg_channels, kernel_size=3, stride=1, padding=1, bias=True)

        # E+V Murged classifier Head
        if self.args.eeg_signal:
            if self.args.gcn:
                # self.eeg_proj = nn.Sequential(
                # nn.Conv2d(eeg_channels, eeg_channels, kernel_size=1),
                # nn.BatchNorm2d(eeg_channels),
                # nn.ReLU()
                # )
                self.eeg_feat_proj = nn.Linear(self.spectrogram_size[0] * self.spectrogram_size[1], self.embed_dim)
                # self.eeg_feat_proj = nn.Conv1d(ch, ch, kernel_size=1)  # across (F*T)

                self.gcn_local = GCN(input_dim=self.embed_dim, output_dim=self.embed_dim, type= "local")
                self.gcn_region = GCN(input_dim=self.embed_dim, output_dim=self.num_classes, type = "region")
                self.region_pool = RegionalAttentionPool(d_model=self.embed_dim)
            
            else:
                # self.classifier = nn.Sequential(
                #     nn.Linear(self.hs, int(self.hs / 2)),
                #     nn.LayerNorm(int(self.hs / 2)),
                #     nn.GELU(),
                #     nn.Linear(int(self.hs / 2), self.num_classes),
                # )
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
        # eeg_raw = F.interpolate(eeg, size=self.spectrogram_size, mode='bilinear', align_corners=False)
        video = x["video"].transpose_(-3, -4)
        
        if self.args.set_eeg_only:
            # eeg = self.eeg2fps(eeg_raw)
            output = self.eeg_model(eeg)

        elif self.args.set_video_only:
            output = self.video_model(video)

        elif self.args.eeg_signal:
            # eeg = self.eeg2fps(eeg_raw)
            eeg_f = self.eeg_model(eeg)
            video_f = self.video_model(video)

            if self.args.gcn:
                # eeg_projected = self.eeg_proj(eeg_raw)
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

                output = self.gcn_region(all_nodes)  # [B, D]

            else:
                fused_f = torch.cat((eeg_f, video_f), dim=1)
                output = self.classifier(fused_f)
                
            output_v = output[:, :self.num_class].unsqueeze(-1)
            output_a = output[:, self.num_class:].unsqueeze(-1)
            output = torch.concat((output_v, output_a), dim=-1)
        
        # new_out_shape = output.shape[:-1] + self.output_dim
        # output = output.view(new_out_shape)

        return output