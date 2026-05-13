import os
import math
import time
from tqdm import tqdm
import torch
import torch.nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

def ddp_allreduce_sum(t: torch.Tensor) -> torch.Tensor:
    """All-reduce sum over all ranks if DDP is initialized."""
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


class PTrainer:
    def __init__(
        self,
        model,
        optimizer,
        lr_scheduler,
        loss_function,
        loss_function_v,
        loss_function_a,
        using_coral,
        fusion,
        align_type,
        weight_ce,
        weight_coral,
        training_dataloader,
        validation_dataloader,
        gpu_id,
        gpu,  # kept for backward-compat; not used
        checkpoint_dir,
        experiment_name="",
        patience=20,
        conf_gate=False,
        stop_gradient=False,
        expert_warmup_epoch=20,
        per_clip_aux_loss=0.0,
    ):
        self.time_stamp = "00000000"
        self.experiment_name = experiment_name
        self.checkpoint_dir = checkpoint_dir

        self.log_dir = os.path.join(self.checkpoint_dir, self.experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.val_log_path = os.path.join(self.log_dir, "val_loss.txt")
        self.train_log_path = os.path.join(self.log_dir, "train_loss.txt")

        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader

        self.model = model
        # for p in self.model.parameters():
        #     p.requires_grad = True
        # for name, p in model.named_parameters():
        #     if p.dtype.is_floating_point:
        #         p.requires_grad = True
        #     else:
        #         p.requires_grad = False

        self.loss_function = loss_function
        self.ce_val = loss_function_v
        self.ce_aro = loss_function_a
        self.CE = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        self.using_coral = using_coral
        self.align_type = align_type

        self.fusion = fusion

        self.w_ce = weight_ce
        self.w_coral = weight_coral

        self.patience = patience
        self.conf_gate = conf_gate
        self.stop_gradient = stop_gradient

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.gpu_id = gpu_id
        self.device = torch.device(f"cuda:{gpu_id}")
        self.is_main = (self.gpu_id == 0)

        self.step = 0
        self.current_epoch = 0
        
        self.best_acc = -1.0
        self.best_epoch = -1

        self.best_metrics = {
            "acc": -1.0,
            "uar": -1.0,
            "f1_macro": -1.0,
            "f1_weighted": -1.0,
        }
        self.best_log_path = os.path.join(self.log_dir, "best.txt")
        self._router_w_accum = []  # [(w_eeg, w_video, w_gcn), ...]

        self.expert_warmup_epoch = expert_warmup_epoch
        self.per_clip_aux_loss_w = per_clip_aux_loss

    def _inner_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _save_checkpoint(self):
        if not self.is_main:
            return
        checkpoint = getattr(self.model, "module", self.model).state_dict()
        ckpt_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        checkpoint_path = f"{ckpt_dir}/{self.time_stamp}_{self.experiment_name}_{self.current_epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Epoch {self.current_epoch}: checkpoint saved at {checkpoint_path}")

    @staticmethod
    def _time_stamp():
        return str(math.floor(time.time()))

    def CORAL_naive(self, source, target):
        d = source.data.shape[1]

        # source covariance
        xm = torch.mean(source, 0, keepdim=True) - source
        xc = xm.t() @ xm

        # target covariance
        xmt = torch.mean(target, 0, keepdim=True) - target
        xct = xmt.t() @ xmt

        # frobenius norm between source and target
        loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
        # loss = loss / (4 * d * d)

        return loss

    def mmd(self, x1, x2, beta):
        x1x1 = self.gaussian_kernel(x1, x1, beta)
        x1x2 = self.gaussian_kernel(x1, x2, beta)
        x2x2 = self.gaussian_kernel(x2, x2, beta)
        diff = x1x1.mean() - 2 * x1x2.mean() + x2x2.mean()
        return diff

    def gaussian_kernel(self, x1, x2, beta = 1.0):
        r = x1.unsqueeze(1)
        diff = r - x2.unsqueeze(0)
        return torch.exp(-beta * (diff ** 2).sum(dim=-1))

    def _disagree_aux_loss(self, logit_eeg, logit_video, target):
        """CE on auxiliary heads — uses class-balanced CE (same as main) to avoid majority bias."""
        if logit_eeg.dim() == 3:
            loss_e = 0.5 * (self.ce_val(logit_eeg[:, :, 0].float(), target[:, 0]) +
                            self.ce_aro(logit_eeg[:, :, 1].float(), target[:, 1]))
            loss_v = 0.5 * (self.ce_val(logit_video[:, :, 0].float(), target[:, 0]) +
                            self.ce_aro(logit_video[:, :, 1].float(), target[:, 1]))
        else:
            loss_e = self.ce_val(logit_eeg.float(), target)
            loss_v = self.ce_val(logit_video.float(), target)
        return 0.2 * (loss_e + loss_v)

    def cmd_loss(self, x, y, k: int = 2, align_mean: bool = False, eps: float = 1e-6):
        if x.dim() == 3:
            x = x.reshape(-1, x.size(-1))
        if y.dim() == 3:
            y = y.reshape(-1, y.size(-1))
        assert x.size(-1) == y.size(-1)

        mx, my = x.mean(0), y.mean(0)
        xc, yc = x - mx, y - my

        sx = torch.sqrt(xc.var(0, unbiased=True) + eps)
        sy = torch.sqrt(yc.var(0, unbiased=True) + eps)
        s = (sx + sy) * 0.5

        mean_term = ((mx - my) / s).pow(2).mean() if align_mean else 0.0

        xc = xc / s
        yc = yc / s

        moments_loss = 0.0
        for order in range(2, k + 1):
            mk_x = (xc.pow(order)).mean(0)
            mk_y = (yc.pow(order)).mean(0)
            moments_loss += (mk_x - mk_y).pow(2).mean()

        loss = mean_term + moments_loss
        
        return loss
    
    def coral_loss(self, x, y):
        if x.dim() == 3:
            x = x.reshape(-1, x.size(-1))
        if y.dim() == 3:
            y = y.reshape(-1, y.size(-1))

        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)

        n_x = x.size(0)
        n_y = y.size(0)

        cov_x = (x.t() @ x) / (n_x - 1)
        cov_y = (y.t() @ y) / (n_y - 1)

        d = cov_x.size(0)
        # loss = ((cov_x - cov_y) ** 2).sum() / (4.0 * d * d)

        # Using correlation matrix
        std_x = torch.sqrt(torch.diag(cov_x) + 1e-6)
        std_y = torch.sqrt(torch.diag(cov_y) + 1e-6)
        corr_x = cov_x / (std_x.unsqueeze(1) * std_x.unsqueeze(0) + 1e-6)
        corr_y = cov_y / (std_y.unsqueeze(1) * std_y.unsqueeze(0) + 1e-6)

        off = 1 - torch.eye(d, device=cov_x.device)
        coral_loss = (( (corr_x - corr_y) * off ) ** 2).sum() / (off.sum() + 1e-6)
        
        mean_loss = torch.mean( (x.mean(0) - y.mean(0)) ** 2 )
        loss = coral_loss + 0.01 * mean_loss
        return coral_loss

    def ddp_allgather_tensor(self, tensor):
        world_size = torch.distributed.get_world_size()

        local_size = torch.tensor([tensor.size(0)], device=tensor.device)
        size_list = [torch.zeros_like(local_size) for _ in range(world_size)]
        torch.distributed.all_gather(size_list, local_size)

        sizes = [int(size.item()) for size in size_list]
        max_size = max(sizes)

        pad_size = max_size - tensor.size(0)
        if pad_size > 0:
            padding = torch.zeros(pad_size, *tensor.shape[1:], device=tensor.device, dtype=tensor.dtype)
            tensor = torch.cat([tensor, padding], dim=0)

        gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
        torch.distributed.all_gather(gather_list, tensor)

        result = []
        for t, size in zip(gather_list, sizes):
            result.append(t[:size])

        return torch.cat(result, dim=0)

    def _run_epoch_single(self, epoch) -> None:
        torch.set_grad_enabled(True)
        self.model.train()

        sampler = getattr(self.training_dataloader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        total_focal_va_loss = 0.0
        total_ce_va_loss = 0.0
        total_combined_loss = 0.0
        total_coral_loss = 0.0

        pbar = tqdm(
            total=len(self.training_dataloader),
            desc=f"Train Epoch {epoch}",
            disable=not self.is_main,
        )

        for batch_number, sample in enumerate(self.training_dataloader):
            # sample = {k: v.to(self.device, non_blocking=True) for k, v in sample.items() if v is not None}
            sample = {
                k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in sample.items()
                if v is not None
            }

            self.optimizer.zero_grad()

            per_clip_logits = None  # set in K-clip set_video_only path when --per_clip_aux_loss > 0

            if self.using_coral:
                if not self.conf_gate:
                    output, eeg_f, video_f = self.model(sample)
                else:
                    output, logit_eeg, logit_video, eeg_f, video_f = self.model(sample)

                if self.align_type == 'coral_naive':
                    coral_loss = self.CORAL_naive(video_f, eeg_f.detach())
                elif self.align_type == 'mmd':
                    coral_loss = self.mmd(video_f, eeg_f.detach(), 1)
                else:
                    coral_loss = self.coral_loss(video_f, eeg_f.detach())

            elif self.fusion == 'router':
                output, gcn_logit, eeg_logit, video_logit, router_logits = self.model(sample)
                if self.is_main and batch_number == 0 and epoch == 0:
                    print(f"[DEBUG shapes] output={tuple(output.shape)} gcn={tuple(gcn_logit.shape)} eeg={tuple(eeg_logit.shape)} vid={tuple(video_logit.shape)} router={tuple(router_logits.shape)}")
                coral_loss = torch.zeros((), device=self.device)
                with torch.no_grad():
                    _rl = router_logits.detach().float()
                    _rls = _rl.size(-1)
                    if _rls in (4, 6):
                        _half = _rls // 2
                        _rw = torch.cat([torch.softmax(_rl[:, :_half], dim=-1),
                                         torch.softmax(_rl[:, _half:], dim=-1)], dim=-1).mean(0).cpu()
                    else:
                        _rw = torch.softmax(_rl, dim=-1).mean(0).cpu()
                    self._router_w_accum.append(_rw)
            elif (self.conf_gate and not self.using_coral) or self.stop_gradient:
                output, logit_eeg, logit_video = self.model(sample)
                coral_loss = torch.zeros((), device=self.device)
            else:
                model_out = self.model(sample)
                # set_video_only K-clip path returns (output, per_clip_logits) when
                # --per_clip_aux_loss > 0; otherwise returns just `output`.
                if isinstance(model_out, tuple):
                    output, per_clip_logits = model_out
                else:
                    output = model_out
                    per_clip_logits = None
                coral_loss = torch.zeros((), device=self.device)

            target = sample["output"]

            if not self.fusion == 'router':
                if output.dim() == 3 and output.size(-1) == 2:
                    # CE
                    v_loss = self.ce_val(output[:, :, 0].float(), target[:, 0])
                    a_loss = self.ce_aro(output[:, :, 1].float(), target[:, 1])
                    ce_va = 0.5 * (v_loss + a_loss)

                    # Focal
                    v_focal_loss = self.loss_function(output[:, :, 0].float(), target[:, 0])
                    a_focal_loss = self.loss_function(output[:, :, 1].float(), target[:, 1])
                    focal_va = 0.5 * (v_focal_loss + a_focal_loss)

                elif output.dim() == 2:
                    ce_va = self.ce_val(output.float(), target)
                    focal_va = self.loss_function(output.float(), target)

                combined = (1-self.w_ce) * focal_va + self.w_ce * ce_va

                # Per-clip auxiliary loss (set_video_only K-clip path only).
                if self.per_clip_aux_loss_w > 0.0 and per_clip_logits is not None:
                    if per_clip_logits.dim() == 4 and per_clip_logits.size(-1) == 2:
                        B_, K_ = per_clip_logits.shape[:2]
                        pcl = per_clip_logits.reshape(B_ * K_, *per_clip_logits.shape[2:])  # [B*K, n_v, 2]
                        tgt_x = target.unsqueeze(1).expand(-1, K_, -1).reshape(B_ * K_, -1)  # [B*K, 2]
                        v_aux = self.ce_val(pcl[:, :, 0].float(), tgt_x[:, 0])
                        a_aux = self.ce_aro(pcl[:, :, 1].float(), tgt_x[:, 1])
                        per_clip_loss = 0.5 * (v_aux + a_aux)
                    else:
                        B_, K_ = per_clip_logits.shape[:2]
                        pcl = per_clip_logits.reshape(B_ * K_, -1)
                        tgt_x = target.unsqueeze(1).expand(-1, K_).reshape(-1)
                        per_clip_loss = self.ce_val(pcl.float(), tgt_x)
                    combined = combined + self.per_clip_aux_loss_w * per_clip_loss
                    if self.is_main and batch_number == 0:
                        print(
                            f"[per_clip_aux] epoch={epoch} K={per_clip_logits.shape[1]} "
                            f"main_combined={combined.item() - (self.per_clip_aux_loss_w * per_clip_loss).item():.4f} "
                            f"per_clip_loss={per_clip_loss.item():.4f} "
                            f"weight={self.per_clip_aux_loss_w} "
                            f"final_combined={combined.item():.4f}"
                        )

            elif self.fusion == 'router':

                # Stage schedule (stop_gradient: warmup → joint)
                if self.stop_gradient and epoch < self.expert_warmup_epoch:
                    # Warmup: train experts; router forward only (no GCN gradient)
                    w_final = 0.0
                    w_e = 1.0
                    w_v = 1.0
                    w_g = 0.0
                    w_router = 0.0
                elif self.stop_gradient:
                    # Joint: train backbones + GCN + router together
                    w_final = 1.0
                    w_e = 1.0
                    w_v = 1.0
                    w_g = 1.0
                    w_router = 0.1
                else:
                    w_final = 1.0
                    w_e = 0.0
                    w_v = 0.0
                    w_g = 0.0
                    w_router = 0.1


                _no_gcn = (router_logits.size(-1) in (2, 4))

                # GCN + stop_gradient also returns log-probs (prob-space mixing)
                _is_prob_output = _no_gcn or (self.stop_gradient and router_logits.size(-1) == 6)

                val_weight = self.ce_val.weight if hasattr(self.ce_val, "weight") else None
                aro_weight = self.ce_aro.weight if hasattr(self.ce_aro, "weight") else None

                # Final router output loss
                if _is_prob_output:
                    # output is log-probability
                    if output.dim() == 3 and output.size(-1) == 2:
                        _v_nll = F.nll_loss(
                            output[:, :, 0].float(),
                            target[:, 0],
                            weight=val_weight
                        )
                        _a_nll = F.nll_loss(
                            output[:, :, 1].float(),
                            target[:, 1],
                            weight=aro_weight
                        )
                        final_loss = 0.5 * (_v_nll + _a_nll)
                    else:
                        final_loss = F.nll_loss(output.float(), target)

                else:
                    # output is raw logit
                    if output.dim() == 3 and output.size(-1) == 2:
                        _v_ce = self.ce_val(output[:, :, 0].float(), target[:, 0])
                        _a_ce = self.ce_aro(output[:, :, 1].float(), target[:, 1])
                        _ce = 0.5 * (_v_ce + _a_ce)

                        _v_focal = self.loss_function(output[:, :, 0].float(), target[:, 0])
                        _a_focal = self.loss_function(output[:, :, 1].float(), target[:, 1])
                        _focal = 0.5 * (_v_focal + _a_focal)

                        final_loss = (1 - self.w_ce) * _focal + self.w_ce * _ce
                    else:
                        _ce = self.ce_val(output.float(), target)
                        _focal = self.loss_function(output.float(), target)
                        final_loss = (1 - self.w_ce) * _focal + self.w_ce * _ce

                # important: final loss is weighted
                combined = w_final * final_loss

                # sample-wise loss (reduction='none')
                if output.dim() == 3 and output.size(-1) == 2:
                    eeg_ce = 0.5 * (
                        F.cross_entropy(eeg_logit[:, :, 0], target[:, 0], weight=val_weight, reduction='none') +
                        F.cross_entropy(eeg_logit[:, :, 1], target[:, 1], weight=aro_weight, reduction='none')
                    )
                    eeg_focal = 0.5 * (
                        self.loss_function(eeg_logit[:, :, 0], target[:, 0]) +
                        self.loss_function(eeg_logit[:, :, 1], target[:, 1])
                    )
                    video_ce = 0.5 * (
                        F.cross_entropy(video_logit[:, :, 0], target[:, 0], weight=val_weight, reduction='none') +
                        F.cross_entropy(video_logit[:, :, 1], target[:, 1], weight=aro_weight, reduction='none')
                    )
                    video_focal = 0.5 * (
                        self.loss_function(video_logit[:, :, 0], target[:, 0]) +
                        self.loss_function(video_logit[:, :, 1], target[:, 1])
                    )
                    if not _no_gcn:
                        n_v = self.model.module.output_dim[0] if hasattr(self.model, "module") else \
                        self.model.output_dim[0]
                        gcn_ce = 0.5 * (
                            F.cross_entropy(gcn_logit[:, :n_v], target[:, 0], weight=val_weight, reduction='none') +
                            F.cross_entropy(gcn_logit[:, n_v:], target[:, 1], weight=aro_weight, reduction='none')
                        )
                        gcn_focal = 0.5 * (
                            self.loss_function(gcn_logit[:, :n_v], target[:, 0]) +
                            self.loss_function(gcn_logit[:, n_v:], target[:, 1])
                        )
                else:
                    eeg_ce = F.cross_entropy(eeg_logit, target, reduction='none')
                    video_ce = F.cross_entropy(video_logit, target, reduction='none')
                    eeg_focal = self.loss_function(eeg_logit, target)
                    video_focal = self.loss_function(video_logit, target)
                    if not _no_gcn:
                        gcn_ce = F.cross_entropy(gcn_logit, target, reduction='none')
                        gcn_focal = self.loss_function(gcn_logit, target)

                # combined
                eeg_combined = (1 - self.w_ce) * eeg_focal + self.w_ce * eeg_ce.mean()
                video_combined = (1 - self.w_ce) * video_focal + self.w_ce * video_ce.mean()
                if not _no_gcn:
                    gcn_combined = (1 - self.w_ce) * gcn_focal + self.w_ce * gcn_ce.mean()

                if self.is_main and batch_number == 0 and epoch == 0:
                    print(f"\n[DEBUG loss] eeg_focal={eeg_focal.item():.4f} eeg_ce={eeg_ce.mean().item():.4f} eeg_combined={eeg_combined.item():.4f}")
                    print(f"[DEBUG loss] vid_focal={video_focal.item():.4f} vid_ce={video_ce.mean().item():.4f} vid_combined={video_combined.item():.4f}")
                    if not _no_gcn:
                        print(f"[DEBUG loss] gcn_focal={gcn_focal.item():.4f} gcn_ce={gcn_ce.mean().item():.4f} gcn_combined={gcn_combined.item():.4f}")
                    print(f"[DEBUG loss] eeg_logit[:,:,0] min={eeg_logit[:,:,0].min().item():.3f} max={eeg_logit[:,:,0].max().item():.3f}")
                    print(f"[DEBUG loss] vid_logit[:,:,0] min={video_logit[:,:,0].min().item():.3f} max={video_logit[:,:,0].max().item():.3f}")
                    if not _no_gcn:
                        print(f"[DEBUG loss] gcn_logit min={gcn_logit.min().item():.3f} max={gcn_logit.max().item():.3f}")
                    print(f"[DEBUG loss] target sample={target[:4].tolist()}\n")

                    focal_va = (1.0 / 3.0) * (
                            eeg_focal.detach() +
                            video_focal.detach() +
                            gcn_focal.detach()
                    )
                    ce_va = (1.0 / 3.0) * (
                            eeg_ce.mean().detach() +
                            video_ce.mean().detach() +
                            gcn_ce.mean().detach()
                    )

                else:
                    focal_va = 0.5 * (
                            eeg_focal.detach() +
                            video_focal.detach()
                    )
                    ce_va = 0.5 * (
                            eeg_ce.mean().detach() +
                            video_ce.mean().detach()
                    )

                # oracle + router loss
                _rls = router_logits.size(-1)
                if _rls in (4, 6) and output.dim() == 3:
                    # V/A separate routing
                    _ec_v = F.cross_entropy(eeg_logit[:, :, 0].float(), target[:, 0],
                                            weight=val_weight, reduction='none').detach()
                    _ec_a = F.cross_entropy(eeg_logit[:, :, 1].float(), target[:, 1],
                                            weight=aro_weight, reduction='none').detach()
                    _vc_v = F.cross_entropy(video_logit[:, :, 0].float(), target[:, 0],
                                            weight=val_weight, reduction='none').detach()
                    _vc_a = F.cross_entropy(video_logit[:, :, 1].float(), target[:, 1],
                                            weight=aro_weight, reduction='none').detach()
                    # _ec_v = F.cross_entropy(eeg_logit[:,:,0], target[:,0], reduction='none').detach()
                    # _ec_a = F.cross_entropy(eeg_logit[:,:,1], target[:,1], reduction='none').detach()
                    # _vc_v = F.cross_entropy(video_logit[:,:,0], target[:,0], reduction='none').detach()
                    # _vc_a = F.cross_entropy(video_logit[:,:,1], target[:,1], reduction='none').detach()
                    # if _no_gcn:
                    #     oracle_v = torch.argmin(torch.stack([_ec_v, _vc_v], dim=1), dim=1)
                    #     oracle_a = torch.argmin(torch.stack([_ec_a, _vc_a], dim=1), dim=1)
                    #     router_loss = (F.cross_entropy(router_logits[:, :2], oracle_v) +
                    #                    F.cross_entropy(router_logits[:, 2:], oracle_a))

                    if _no_gcn:
                        tau = 0.5

                        ce_pair_v = torch.stack([_ec_v, _vc_v], dim=1)  # [B, 2]
                        ce_pair_a = torch.stack([_ec_a, _vc_a], dim=1)  # [B, 2]

                        target_w_v = torch.softmax(-ce_pair_v / tau, dim=1).detach()
                        target_w_a = torch.softmax(-ce_pair_a / tau, dim=1).detach()

                        log_router_v = F.log_softmax(router_logits[:, :2], dim=1)
                        log_router_a = F.log_softmax(router_logits[:, 2:], dim=1)

                        router_loss_v = F.kl_div(log_router_v, target_w_v, reduction='batchmean')
                        router_loss_a = F.kl_div(log_router_a, target_w_a, reduction='batchmean')

                        router_loss = router_loss_v + router_loss_a

                    else:
                        _gc_v = F.cross_entropy(gcn_logit[:,:5].float(), target[:,0],
                                                weight=val_weight, reduction='none').detach()
                        _gc_a = F.cross_entropy(gcn_logit[:,5:].float(), target[:,1],
                                                weight=aro_weight, reduction='none').detach()
                        if self.stop_gradient:
                            tau = 0.5
                            ce_pair_v = torch.stack([_ec_v, _vc_v, _gc_v], dim=1)
                            ce_pair_a = torch.stack([_ec_a, _vc_a, _gc_a], dim=1)
                            target_w_v = torch.softmax(-ce_pair_v / tau, dim=1).detach()
                            target_w_a = torch.softmax(-ce_pair_a / tau, dim=1).detach()
                            log_router_v = F.log_softmax(router_logits[:, :3], dim=1)
                            log_router_a = F.log_softmax(router_logits[:, 3:], dim=1)
                            router_loss = (F.kl_div(log_router_v, target_w_v, reduction='batchmean') +
                                           F.kl_div(log_router_a, target_w_a, reduction='batchmean'))
                        else:
                            oracle_v = torch.argmin(torch.stack([_ec_v, _vc_v, _gc_v], dim=1), dim=1)
                            oracle_a = torch.argmin(torch.stack([_ec_a, _vc_a, _gc_a], dim=1), dim=1)
                            router_loss = (F.cross_entropy(router_logits[:, :3], oracle_v) +
                                           F.cross_entropy(router_logits[:, 3:], oracle_a))
                else:
                    if _no_gcn:
                        oracle = torch.argmin(torch.stack([eeg_ce.detach(), video_ce.detach()], dim=1), dim=1)
                    else:
                        oracle = torch.argmin(torch.stack([eeg_ce.detach(), video_ce.detach(), gcn_ce.detach()], dim=1), dim=1)
                    router_loss = F.cross_entropy(router_logits, oracle)

                # 4. final loss
                if _no_gcn:
                    combined = combined + w_e * eeg_combined + w_v * video_combined
                else:
                    combined = combined + w_e * eeg_combined + w_v * video_combined + w_g * gcn_combined
                combined = combined + w_router * router_loss

            if (self.conf_gate or self.stop_gradient) and self.fusion != 'router':
                # aux CE on heads to keep them informative as confidence estimators
                # (router already includes w_e * eeg_combined + w_v * video_combined)
                aux = self._disagree_aux_loss(logit_eeg, logit_video, target)
                combined = combined + 0.5 * aux

            if self.using_coral:
                combined = combined + self.w_coral * coral_loss

            loss = combined
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            total_focal_va_loss += focal_va.item()
            total_ce_va_loss += ce_va.item()
            total_combined_loss += combined.item()
            total_coral_loss += coral_loss.item()

            avg_focal_va = total_focal_va_loss / (batch_number + 1)
            avg_ce_va = total_ce_va_loss / (batch_number + 1)
            avg_combined = total_combined_loss / (batch_number + 1)
            avg_coral = total_coral_loss / (batch_number + 1)

            if self.is_main:
                pbar.set_postfix(
                    C=f"{avg_coral:.4f}",
                    F=f"{avg_focal_va:.4f}",
                    CE=f"{avg_ce_va:.4f}",
                    T=f"{avg_combined:.4f}",
                    lr=f'{self.optimizer.param_groups[0]["lr"]:.6f}'
                )
                pbar.update(1)

        pbar.close()

        if self.is_main and self.fusion == 'router' and self._router_w_accum:
            avg_w = torch.stack(self._router_w_accum).mean(0)
            n = avg_w.numel()
            if n == 6:
                print(f"  [router/train] epoch {self.current_epoch}: "
                      f"V: w_eeg={avg_w[0]:.3f} w_video={avg_w[1]:.3f} w_gcn={avg_w[2]:.3f} | "
                      f"A: w_eeg={avg_w[3]:.3f} w_video={avg_w[4]:.3f} w_gcn={avg_w[5]:.3f}")
            elif n == 4:
                print(f"  [router/train] epoch {self.current_epoch}: "
                      f"V: w_eeg={avg_w[0]:.3f} w_video={avg_w[1]:.3f} | "
                      f"A: w_eeg={avg_w[2]:.3f} w_video={avg_w[3]:.3f}")
            elif n == 2:
                print(f"  [router/train] epoch {self.current_epoch}: w_eeg={avg_w[0]:.3f} w_video={avg_w[1]:.3f}")
            else:
                print(f"  [router/train] epoch {self.current_epoch}: w_eeg={avg_w[0]:.3f} w_video={avg_w[1]:.3f} w_gcn={avg_w[2]:.3f}")
        self._router_w_accum.clear()

        with open(self.train_log_path, "a") as f:
            f.write(
                f"{self.current_epoch},"
                f"{avg_coral:.6f},{avg_focal_va:.6f},{avg_ce_va:.6f},{avg_combined:.6f}\n"
            )


    def _eval(self) -> None:
        self.model.eval()
        num_batches = len(self.validation_dataloader)

        total_samples = torch.tensor(0.0, device=self.device)
        total_correct_val = torch.tensor(0.0, device=self.device)
        total_correct_aro = torch.tensor(0.0, device=self.device)

        total_correct_single = torch.tensor(0.0, device=self.device)

        total_focal_va_loss = torch.tensor(0.0, device=self.device)
        total_ce_va_loss    = torch.tensor(0.0, device=self.device)
        total_combined_loss = torch.tensor(0.0, device=self.device)  # focal_va + ce_va

        total_coral_loss    = torch.tensor(0.0, device=self.device)

        total_correct_eeg_val = torch.tensor(0.0, device=self.device)
        total_correct_eeg_aro = torch.tensor(0.0, device=self.device)
        total_correct_video_val = torch.tensor(0.0, device=self.device)
        total_correct_video_aro = torch.tensor(0.0, device=self.device)
        total_correct_gcn_val = torch.tensor(0.0, device=self.device)
        total_correct_gcn_aro = torch.tensor(0.0, device=self.device)
        gcn_seen = False  # set True when GCN logits are available in this run

        all_preds_val, all_targets_val = [], []
        all_preds_aro, all_targets_aro = [], []

        all_preds_single, all_targets_single = [], []

        pbar = tqdm(total=num_batches, desc=f"Validate Epoch {self.current_epoch}", disable=not self.is_main)

        correct_paths = []
        wrong_paths = []
        router_records = []  # per-sample router analysis
        _eval_no_gcn = False    # set True when no-GCN router (router_logits size in (2,4))
        _eval_is_prob = False   # set True when output is log-probs (no-GCN OR gcn+stop_gradient)

        router_w_sum = None
        router_w_count = torch.tensor(0.0, device=self.device)

        with torch.no_grad():
            for _, sample in enumerate(self.validation_dataloader):
                paths = sample.get("path", None)

                # sample = {k: v.to(self.device, non_blocking=True) for k, v in sample.items() if v is not None}
                sample = {
                    k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v)
                    for k, v in sample.items()
                    if v is not None
                }
                target = sample["output"]
                bs = target.size(0)

                if self.using_coral:
                    if not self.conf_gate:
                        preds, eeg_f, video_f = self.model(sample)
                    elif self.conf_gate:
                        preds, logit_eeg, logit_video, eeg_f, video_f = self.model(sample)

                    if self.align_type == 'coral_naive':
                        coral_loss = self.CORAL_naive(video_f, eeg_f.detach())
                    elif self.align_type == 'mmd':
                        coral_loss = self.mmd(video_f, eeg_f.detach(), 1)
                    else:
                        coral_loss = self.coral_loss(video_f, eeg_f.detach())

                elif self.fusion == 'router':
                    preds, _gcn_l, _eeg_l, _vid_l, _rlogits = self.model(sample)
                    coral_loss = torch.zeros((), device=self.device)

                    # ============================================================
                    # Branch-wise expert accuracy for best expert checkpoint
                    # ============================================================
                    if _eeg_l.dim() == 3 and _eeg_l.size(-1) == 2:
                        _eeg_pred = _eeg_l.argmax(1)  # [B, 2]
                        total_correct_eeg_val += (_eeg_pred[:, 0] == target[:, 0]).sum()
                        total_correct_eeg_aro += (_eeg_pred[:, 1] == target[:, 1]).sum()

                    if _vid_l.dim() == 3 and _vid_l.size(-1) == 2:
                        _vid_pred = _vid_l.argmax(1)  # [B, 2]
                        total_correct_video_val += (_vid_pred[:, 0] == target[:, 0]).sum()
                        total_correct_video_aro += (_vid_pred[:, 1] == target[:, 1]).sum()

                    # GCN accuracy: gcn_logit is flat [B, n_v+n_a] when GCN is active.
                    if (_gcn_l is not None) and _gcn_l.dim() == 2:
                        inner = self.model.module if hasattr(self.model, "module") else self.model
                        n_v = inner.output_dim[0] if hasattr(inner, "output_dim") else (_gcn_l.size(-1) // 2)
                        if _gcn_l.size(-1) >= n_v * 2:
                            gcn_seen = True
                            _gcn_pred_v = _gcn_l[:, :n_v].argmax(1)
                            _gcn_pred_a = _gcn_l[:, n_v:].argmax(1)
                            total_correct_gcn_val += (_gcn_pred_v == target[:, 0]).sum()
                            total_correct_gcn_aro += (_gcn_pred_a == target[:, 1]).sum()

                    _rl = _rlogits.detach().float()
                    _rls = _rl.size(-1)
                    _eval_no_gcn = (_rls in (2, 4))
                    _eval_is_prob = _eval_no_gcn or (self.stop_gradient and _rls == 6)
                    # if _rls in (4, 6):
                    #     _half = _rls // 2
                    #     _rw_batch = torch.cat([torch.softmax(_rl[:, :_half], dim=-1),
                    #                            torch.softmax(_rl[:, _half:], dim=-1)], dim=-1).cpu()
                    # else:
                    #     _rw_batch = torch.softmax(_rl, dim=-1).cpu()
                    # self._router_w_accum.append(_rw_batch.mean(0))

                    if _rls in (4, 6):
                        _half = _rls // 2
                        _rw_batch_dev = torch.cat([
                            torch.softmax(_rl[:, :_half], dim=-1),
                            torch.softmax(_rl[:, _half:], dim=-1)
                        ], dim=-1)  # [B, K], device
                    else:
                        _rw_batch_dev = torch.softmax(_rl, dim=-1)  # [B, K], device

                    if router_w_sum is None:
                        router_w_sum = _rw_batch_dev.sum(dim=0)
                    else:
                        router_w_sum += _rw_batch_dev.sum(dim=0)

                    router_w_count += _rw_batch_dev.size(0)

                    _rw_batch = _rw_batch_dev.cpu()

                    # per-sample router analysis (all samples, all batches)
                    if preds.dim() == 3:
                        _rls = _rw_batch.size(-1)
                        _no_gcn_rec = (_rls in (2, 4))
                        _is_va_rec = (_rls in (4, 6))
                        _rw_np = _rw_batch.numpy()

                        def _pred_va_from_logits(logits):
                            """
                            logits:
                                [B, C, 2]  -> return [B, 2]
                                [B, 2C]    -> return [B, 2]
                            """
                            if logits.dim() == 3 and logits.size(-1) == 2:
                                return logits.argmax(1)  # [B, 2]

                            elif logits.dim() == 2:
                                n_v = target.size(1)  # this is 2, not class num, so don't use this
                                # Use class number from prediction shape.
                                # For emognition/mdmer, flat shape is [B, 2C].
                                c = logits.size(1) // 2
                                pred_v = logits[:, :c].argmax(1)
                                pred_a = logits[:, c:].argmax(1)
                                return torch.stack([pred_v, pred_a], dim=1)  # [B, 2]

                            else:
                                raise ValueError(f"Unexpected logits shape for V/A prediction: {logits.shape}")

                        _p_eeg = _pred_va_from_logits(_eeg_l).cpu().numpy()
                        _p_vid = _pred_va_from_logits(_vid_l).cpu().numpy()
                        _p_fin = _pred_va_from_logits(preds).cpu().numpy()
                        _tgt = target.cpu().numpy()

                        if not _no_gcn_rec:
                            _p_gcn = _pred_va_from_logits(_gcn_l).cpu().numpy()

                        for i in range(target.size(0)):
                            rec = {
                                "target_v": int(_tgt[i, 0]),
                                "target_a": int(_tgt[i, 1]),

                                "pred_eeg_v": int(_p_eeg[i, 0]),
                                "pred_eeg_a": int(_p_eeg[i, 1]),

                                "pred_vid_v": int(_p_vid[i, 0]),
                                "pred_vid_a": int(_p_vid[i, 1]),

                                "pred_fin_v": int(_p_fin[i, 0]),
                                "pred_fin_a": int(_p_fin[i, 1]),

                                "eeg_ok": bool(_p_eeg[i, 0] == _tgt[i, 0] and _p_eeg[i, 1] == _tgt[i, 1]),
                                "vid_ok": bool(_p_vid[i, 0] == _tgt[i, 0] and _p_vid[i, 1] == _tgt[i, 1]),
                                "fin_ok": bool(_p_fin[i, 0] == _tgt[i, 0] and _p_fin[i, 1] == _tgt[i, 1]),
                            }

                            if not _no_gcn_rec:
                                rec.update({
                                    "pred_gcn_v": int(_p_gcn[i, 0]),
                                    "pred_gcn_a": int(_p_gcn[i, 1]),
                                    "gcn_ok": bool(_p_gcn[i, 0] == _tgt[i, 0] and _p_gcn[i, 1] == _tgt[i, 1]),
                                })

                            if _is_va_rec:
                                if _no_gcn_rec:
                                    rec.update({
                                        "w_eeg_v": float(_rw_np[i, 0]),
                                        "w_video_v": float(_rw_np[i, 1]),
                                        "w_eeg_a": float(_rw_np[i, 2]),
                                        "w_video_a": float(_rw_np[i, 3]),
                                    })
                                else:
                                    rec.update({
                                        "w_eeg_v": float(_rw_np[i, 0]),
                                        "w_video_v": float(_rw_np[i, 1]),
                                        "w_gcn_v": float(_rw_np[i, 2]),
                                        "w_eeg_a": float(_rw_np[i, 3]),
                                        "w_video_a": float(_rw_np[i, 4]),
                                        "w_gcn_a": float(_rw_np[i, 5]),
                                    })
                            else:
                                if _no_gcn_rec:
                                    rec.update({
                                        "w_eeg": float(_rw_np[i, 0]),
                                        "w_video": float(_rw_np[i, 1]),
                                    })
                                else:
                                    rec.update({
                                        "w_eeg": float(_rw_np[i, 0]),
                                        "w_video": float(_rw_np[i, 1]),
                                        "w_gcn": float(_rw_np[i, 2]),
                                    })

                            router_records.append(rec)
                elif self.conf_gate or self.stop_gradient:
                    preds, _, _ = self.model(sample)
                    coral_loss = torch.zeros((), device=self.device)
                else:
                    model_out = self.model(sample)
                    # set_video_only K-clip path may return (preds, per_clip_logits)
                    # when --per_clip_aux_loss > 0; eval only needs the pooled preds.
                    if isinstance(model_out, tuple):
                        preds = model_out[0]
                    else:
                        preds = model_out
                    coral_loss = torch.zeros((), device=self.device)

                if preds.dim() == 3 and preds.size(-1) == 2:
                    if _eval_is_prob:
                        # output is log-probs from probability-space mixing
                        val_weight = self.ce_val.weight if hasattr(self.ce_val, "weight") else None
                        aro_weight = self.ce_aro.weight if hasattr(self.ce_aro, "weight") else None

                        _v_nll = F.nll_loss(
                            preds[:, :, 0].float(),
                            target[:, 0],
                            weight=val_weight
                        )
                        _a_nll = F.nll_loss(
                            preds[:, :, 1].float(),
                            target[:, 1],
                            weight=aro_weight
                        )

                        focal_va = ce_va = 0.5 * (_v_nll + _a_nll)

                    else:
                        focal_v = self.loss_function(preds[:, :, 0].float(), target[:, 0])
                        focal_a = self.loss_function(preds[:, :, 1].float(), target[:, 1])
                        focal_va = 0.5 * (focal_v + focal_a)

                        ce_v = self.ce_val(preds[:, :, 0].float(), target[:, 0])
                        ce_a = self.ce_aro(preds[:, :, 1].float(), target[:, 1])
                        ce_va = 0.5 * (ce_v + ce_a)

                    combined = (1 - self.w_ce) * focal_va + self.w_ce * ce_va
                    if self.using_coral:
                        combined = combined + self.w_coral * coral_loss

                    total_focal_va_loss += focal_va * bs
                    total_ce_va_loss    += ce_va    * bs
                    total_coral_loss    += coral_loss * bs
                    total_combined_loss += combined * bs

                    pred = preds.argmax(1)  # [B, 2]
                    total_correct_val += (pred[:, 0] == target[:, 0]).sum()
                    total_correct_aro += (pred[:, 1] == target[:, 1]).sum()

                    all_preds_val.extend(pred[:, 0].detach().cpu().numpy())
                    all_targets_val.extend(target[:, 0].detach().cpu().numpy())
                    all_preds_aro.extend(pred[:, 1].detach().cpu().numpy())
                    all_targets_aro.extend(target[:, 1].detach().cpu().numpy())

                elif preds.dim() == 2:
                    focal_va = self.loss_function(preds.float(), target)
                    ce_va    = self.ce_val(preds.float(), target)

                    combined = (1-self.w_ce) * focal_va + self.w_ce * ce_va
                    # combined = focal_va + ce_va
                    if self.coral_loss:
                        combined = combined + 0.01 * coral_loss

                    total_focal_va_loss += focal_va * bs
                    total_ce_va_loss    += ce_va    * bs
                    total_coral_loss    += coral_loss    * bs
                    total_combined_loss += combined * bs

                    pred = preds.argmax(1)  # [B]
                    total_correct_single += (pred == target).sum()

                    if paths is not None:
                        for p, y_pred, y_gt in zip(paths, pred.cpu().numpy(), target.cpu().numpy()):
                            if y_pred == y_gt:
                                correct_paths.append(p)
                            else:
                                wrong_paths.append(p)

                    all_preds_single.extend(pred.detach().cpu().numpy())
                    all_targets_single.extend(target.detach().cpu().numpy())

                else:
                    raise ValueError(f"Unexpected predictions shape for loss: {preds.shape}")

                total_samples += bs

                if self.is_main:
                    denom = torch.clamp(total_samples, min=1.0)
                    avg_focal_va = (total_focal_va_loss / denom).item()
                    avg_ce_va    = (total_ce_va_loss    / denom).item()
                    avg_coral    = (total_coral_loss    / denom).item()
                    avg_combined = (total_combined_loss / denom).item()

                    # MDMER, Emognition: V/A acc, EAV:ACC
                    if len(all_preds_val) > 0 or len(all_preds_aro) > 0:
                        val_acc_cur  = (total_correct_val / denom).item()
                        aro_acc_cur  = (total_correct_aro / denom).item()
                        pbar.set_postfix(
                            F=f"{avg_focal_va:.4f}",
                            CE=f"{avg_ce_va:.4f}",
                            C=f"{avg_coral:.4f}",
                            T=f"{avg_combined:.4f}",
                            VACC=f"{val_acc_cur:.4f}",
                            AACC=f"{aro_acc_cur:.4f}"
                        )
                    else:
                        acc_cur = (total_correct_single / denom).item()
                        pbar.set_postfix(
                            F=f"{avg_focal_va:.4f}",
                            CE=f"{avg_ce_va:.4f}",
                            C=f"{avg_coral:.4f}",
                            T=f"{avg_combined:.4f}",
                            ACC=f"{acc_cur:.4f}"
                        )
                    pbar.update(1)

        pbar.close()

        # DDP reduce
        total_samples = ddp_allreduce_sum(total_samples)
        total_correct_val = ddp_allreduce_sum(total_correct_val)
        total_correct_aro = ddp_allreduce_sum(total_correct_aro)
        total_correct_single = ddp_allreduce_sum(total_correct_single)
        total_focal_va_loss = ddp_allreduce_sum(total_focal_va_loss)
        total_ce_va_loss = ddp_allreduce_sum(total_ce_va_loss)
        total_coral_loss     = ddp_allreduce_sum(total_coral_loss)
        total_combined_loss = ddp_allreduce_sum(total_combined_loss)

        total_correct_eeg_val = ddp_allreduce_sum(total_correct_eeg_val)
        total_correct_eeg_aro = ddp_allreduce_sum(total_correct_eeg_aro)
        total_correct_gcn_val = ddp_allreduce_sum(total_correct_gcn_val)
        total_correct_gcn_aro = ddp_allreduce_sum(total_correct_gcn_aro)
        total_correct_video_val = ddp_allreduce_sum(total_correct_video_val)
        total_correct_video_aro = ddp_allreduce_sum(total_correct_video_aro)

        # 🔥 list → tensor 변환
        if len(all_preds_val) > 0:
            all_preds_val = torch.tensor(all_preds_val, device=self.device)
            all_targets_val = torch.tensor(all_targets_val, device=self.device)

            all_preds_val = self.ddp_allgather_tensor(all_preds_val)
            all_targets_val = self.ddp_allgather_tensor(all_targets_val)

            all_preds_val = all_preds_val.cpu().numpy()
            all_targets_val = all_targets_val.cpu().numpy()

        if len(all_preds_aro) > 0:
            all_preds_aro = torch.tensor(all_preds_aro, device=self.device)
            all_targets_aro = torch.tensor(all_targets_aro, device=self.device)

            all_preds_aro = self.ddp_allgather_tensor(all_preds_aro)
            all_targets_aro = self.ddp_allgather_tensor(all_targets_aro)

            all_preds_aro = all_preds_aro.cpu().numpy()
            all_targets_aro = all_targets_aro.cpu().numpy()

        if len(all_preds_single) > 0:
            all_preds_single = torch.tensor(all_preds_single, device=self.device)
            all_targets_single = torch.tensor(all_targets_single, device=self.device)

            all_preds_single = self.ddp_allgather_tensor(all_preds_single)
            all_targets_single = self.ddp_allgather_tensor(all_targets_single)

            all_preds_single = all_preds_single.cpu().numpy()
            all_targets_single = all_targets_single.cpu().numpy()

        # -------------------------------------------------------
        # Router analysis / router weight aggregation
        # Must be done BEFORE "if not self.is_main: return"
        # because DDP collectives must be called by all ranks.
        # -------------------------------------------------------
        avg_router_w = None

        if self.fusion == 'router':
            # 1) Aggregate router weight sum/count across all ranks
            if router_w_sum is not None:
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(router_w_sum, op=dist.ReduceOp.SUM)
                    dist.all_reduce(router_w_count, op=dist.ReduceOp.SUM)

                avg_router_w = router_w_sum / router_w_count.clamp(min=1.0)

            # 2) Gather per-sample router records across all ranks
            if dist.is_available() and dist.is_initialized():
                all_records = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(all_records, router_records)
                router_records = [r for recs in all_records for r in recs]

            # 3) Save jsonl only on rank 0
            if self.is_main and router_records:
                ra_dir = os.path.join(self.log_dir, "router_analysis")
                os.makedirs(ra_dir, exist_ok=True)
                ra_path = os.path.join(ra_dir, f"epoch_{self.current_epoch}.jsonl")
                with open(ra_path, "w") as f:
                    for rec in router_records:
                        f.write(json.dumps(rec) + "\n")

        if not self.is_main:
            return

        denom = torch.clamp(total_samples, min=1.0)
        avg_focal_va = (total_focal_va_loss / denom).item()
        avg_ce_va    = (total_ce_va_loss    / denom).item()
        avg_coral    = (total_coral_loss    / denom).item() 
        avg_combined = (total_combined_loss / denom).item()

        is_multihead = (len(all_targets_val) > 0) or (len(all_targets_aro) > 0)

        if is_multihead: # MDMER, Emognition
            val_acc = (total_correct_val / denom).item()
            aro_acc = (total_correct_aro / denom).item()

            # UAR/F1
            val_uar = balanced_accuracy_score(all_targets_val, all_preds_val) if len(all_targets_val)>0 else 0.0
            aro_uar = balanced_accuracy_score(all_targets_aro, all_preds_aro) if len(all_targets_aro)>0 else 0.0

            val_f1_macro    = f1_score(all_targets_val, all_preds_val, average="macro",    zero_division=0) if len(all_targets_val)>0 else 0.0
            val_f1_weighted = f1_score(all_targets_val, all_preds_val, average="weighted", zero_division=0) if len(all_targets_val)>0 else 0.0
            aro_f1_macro    = f1_score(all_targets_aro, all_preds_aro, average="macro",    zero_division=0) if len(all_targets_aro)>0 else 0.0
            aro_f1_weighted = f1_score(all_targets_aro, all_preds_aro, average="weighted", zero_division=0) if len(all_targets_aro)>0 else 0.0

            mean_uar         = 0.5 * (val_uar + aro_uar)
            mean_f1_macro    = 0.5 * (val_f1_macro + aro_f1_macro)
            mean_f1_weighted = 0.5 * (val_f1_weighted + aro_f1_weighted)

            tqdm.write(
                f"[Val][Epoch {self.current_epoch}] "
                f"UAR(V/A/mean)={val_uar:.4f}/{aro_uar:.4f}/{mean_uar:.4f} | "
                f"F1-m(V/A/mean)={val_f1_macro:.4f}/{aro_f1_macro:.4f}/{mean_f1_macro:.4f} | "
                f"F1-w(V/A/mean)={val_f1_weighted:.4f}/{aro_f1_weighted:.4f}/{mean_f1_weighted:.4f} "
            )

            if self.fusion == 'router' and avg_router_w is not None:
                avg_w = avg_router_w.detach().cpu()
                n = avg_w.numel()

                if n == 6:
                    tqdm.write(
                        f"  [router/val]   epoch {self.current_epoch}: "
                        f"V: w_eeg={avg_w[0]:.3f} w_video={avg_w[1]:.3f} w_gcn={avg_w[2]:.3f} | "
                        f"A: w_eeg={avg_w[3]:.3f} w_video={avg_w[4]:.3f} w_gcn={avg_w[5]:.3f}"
                    )
                elif n == 4:
                    tqdm.write(
                        f"  [router/val]   epoch {self.current_epoch}: "
                        f"V: w_eeg={avg_w[0]:.3f} w_video={avg_w[1]:.3f} | "
                        f"A: w_eeg={avg_w[2]:.3f} w_video={avg_w[3]:.3f}"
                    )
                elif n == 2:
                    tqdm.write(
                        f"  [router/val]   epoch {self.current_epoch}: "
                        f"w_eeg={avg_w[0]:.3f} w_video={avg_w[1]:.3f}"
                    )
                elif n == 3:
                    tqdm.write(
                        f"  [router/val]   epoch {self.current_epoch}: "
                        f"w_eeg={avg_w[0]:.3f} w_video={avg_w[1]:.3f} w_gcn={avg_w[2]:.3f}"
                    )
                else:
                    tqdm.write(
                        f"  [router/val]   epoch {self.current_epoch}: avg_w={avg_w.tolist()}"
                    )


            with open(self.val_log_path, "a") as f:
                f.write(
                    f"{self.current_epoch},"
                    f"{avg_focal_va:.6f},{avg_ce_va:.6f},{avg_combined:.6f},"
                    f"{avg_coral:.6f},"
                    f"{val_acc:.6f},{aro_acc:.6f},"
                    f"{val_uar:.6f},{aro_uar:.6f},"
                    f"{val_f1_macro:.6f},{aro_f1_macro:.6f},"
                    f"{val_f1_weighted:.6f},{aro_f1_weighted:.6f}\n"
                )

            # cm_val = confusion_matrix(all_targets_val, all_preds_val)
            # cm_aro = confusion_matrix(all_targets_aro, all_preds_aro)
            # cm_dir = os.path.join(self.log_dir, "confusion_matrix")
            # os.makedirs(cm_dir, exist_ok=True)

            # # Valence confusion matrix
            # plt.figure(figsize=(6, 5))
            # sns.heatmap(cm_val, annot=True, fmt="d", cmap="Blues")
            # plt.title(f"Valence Confusion Matrix (Epoch {self.current_epoch})")
            # plt.xlabel("Predicted")
            # plt.ylabel("True")
            # plt.tight_layout()
            # plt.savefig(os.path.join(cm_dir, f"val_epoch{self.current_epoch}.png"))
            # plt.close()

            # # Arousal confusion matrix
            # plt.figure(figsize=(6, 5))
            # sns.heatmap(cm_aro, annot=True, fmt="d", cmap="Blues")
            # plt.title(f"Arousal Confusion Matrix (Epoch {self.current_epoch})")
            # plt.xlabel("Predicted")
            # plt.ylabel("True")
            # plt.tight_layout()
            # plt.savefig(os.path.join(cm_dir, f"aro_epoch{self.current_epoch}.png"))
            # plt.close()

            # ------------------------------------------------------------
            # Branch expert metrics
            # ------------------------------------------------------------
            eeg_val_acc = (total_correct_eeg_val / denom).item()
            eeg_aro_acc = (total_correct_eeg_aro / denom).item()
            video_val_acc = (total_correct_video_val / denom).item()
            video_aro_acc = (total_correct_video_aro / denom).item()
            gcn_val_acc = (total_correct_gcn_val / denom).item()
            gcn_aro_acc = (total_correct_gcn_aro / denom).item()

            eeg_mean_acc = 0.5 * (eeg_val_acc + eeg_aro_acc)
            video_mean_acc = 0.5 * (video_val_acc + video_aro_acc)
            gcn_mean_acc = 0.5 * (gcn_val_acc + gcn_aro_acc)

            if self.fusion == "router":
                _gcn_log_line = ""
                if gcn_seen:
                    _gcn_log_line = (
                        f" | GCN acc(V/A/mean)={gcn_val_acc:.4f}/{gcn_aro_acc:.4f}/{gcn_mean_acc:.4f}"
                    )
                tqdm.write(
                    f"  [expert/val] epoch {self.current_epoch}: "
                    f"EEG acc(V/A/mean)={eeg_val_acc:.4f}/{eeg_aro_acc:.4f}/{eeg_mean_acc:.4f} | "
                    f"Video acc(V/A/mean)={video_val_acc:.4f}/{video_aro_acc:.4f}/{video_mean_acc:.4f}"
                    f"{_gcn_log_line}"
                )

            return {
                "acc": (val_acc + aro_acc) / 2,
                "uar": mean_uar,
                "f1_macro": mean_f1_macro,
                "f1_weighted": mean_f1_weighted,

                "eeg_acc": eeg_mean_acc,
                "eeg_val_acc": eeg_val_acc,
                "eeg_aro_acc": eeg_aro_acc,
                "video_acc": video_mean_acc,
                "video_val_acc": video_val_acc,
                "video_aro_acc": video_aro_acc,
                "gcn_acc": gcn_mean_acc,
                "gcn_val_acc": gcn_val_acc,
                "gcn_aro_acc": gcn_aro_acc,
            }

        else: # EAV
            acc_single = (total_correct_single / denom).item()

            if len(all_targets_single) > 0:
                uar_single = balanced_accuracy_score(all_targets_single, all_preds_single)
                f1m_single = f1_score(all_targets_single, all_preds_single, average="macro",    zero_division=0)
                f1w_single = f1_score(all_targets_single, all_preds_single, average="weighted", zero_division=0)
            else:
                uar_single = f1m_single = f1w_single = 0.0

            tqdm.write(
                f"[Val][Epoch {self.current_epoch}] "
                f"ACC={acc_single:.4f} | UAR={uar_single:.4f} | F1-m={f1m_single:.4f} | F1-w={f1w_single:.4f}"
            )

            with open(self.val_log_path, "a") as f:
                f.write(
                    f"{self.current_epoch},"
                    f"{avg_focal_va:.6f},{avg_ce_va:.6f},{avg_combined:.6f},"
                    f"{avg_coral:.6f},"
                    f"{acc_single:.6f},{0.0:.6f},"
                    f"{uar_single:.6f},{uar_single:.6f},"
                    f"{f1m_single:.6f},{f1m_single:.6f},"
                    f"{f1w_single:.6f},{f1w_single:.6f}\n"
                )
            self.last_acc = acc_single

            if self.is_main:
                save_dir = os.path.join(self.log_dir, "eval_samples")
                os.makedirs(save_dir, exist_ok=True)

                with open(os.path.join(save_dir, f"correct_epoch_{self.current_epoch}.txt"), "w") as f:
                    for p in correct_paths:
                        f.write(f"{p}\n")

                with open(os.path.join(save_dir, f"wrong_epoch_{self.current_epoch}.txt"), "w") as f:
                    for p in wrong_paths:
                        f.write(f"{p}\n")

            return {
                "acc": acc_single,
                "uar": uar_single,
                "f1_macro": f1m_single,
                "f1_weighted": f1w_single,
            }

    def save_test_predictions(self, model_name: str, save_dir: str):
        self.model.eval()
        os.makedirs(save_dir, exist_ok=True)

        jsonl_path = os.path.join(save_dir, f"{model_name}_predictions.jsonl")
        txt_path = os.path.join(save_dir, f"{model_name}_predictions.txt")

        num_batches = len(self.validation_dataloader)
        pbar = tqdm(total=num_batches, desc=f"Test-{model_name}", disable=not self.is_main)

        records = []

        with torch.no_grad():
            for _, sample in enumerate(self.validation_dataloader):
                raw_paths = sample.get("path", None)
                raw_eeg_paths = sample.get("eeg_path", None)
                raw_indices = sample.get("index", None)

                sample = {
                    k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v)
                    for k, v in sample.items()
                    if v is not None
                }

                target = sample["output"]
                result = self.model(sample)
                preds = result[0] if isinstance(result, tuple) else result

                if preds.dim() == 3 and preds.size(-1) == 2:
                    pred = preds.argmax(1)  # [B, 2]

                    val_correct = (pred[:, 0] == target[:, 0])
                    aro_correct = (pred[:, 1] == target[:, 1])
                    joint_correct = val_correct & aro_correct

                    bs = target.size(0)
                    for i in range(bs):
                        rec = {
                            "index": int(raw_indices[i]) if raw_indices is not None and not torch.is_tensor(raw_indices)
                            else (int(raw_indices[i].item()) if raw_indices is not None else i),
                            "path": raw_paths[i] if raw_paths is not None else None,
                            "eeg_path": raw_eeg_paths[i] if raw_eeg_paths is not None else None,
                            "target": target[i].detach().cpu().tolist(),
                            "pred": pred[i].detach().cpu().tolist(),
                            "val_correct": bool(val_correct[i].item()),
                            "aro_correct": bool(aro_correct[i].item()),
                            "joint_correct": bool(joint_correct[i].item()),
                        }
                        records.append(rec)

                elif preds.dim() == 2:
                    pred = preds.argmax(1)
                    correct = (pred == target)

                    bs = target.size(0)
                    for i in range(bs):
                        rec = {
                            "index": int(raw_indices[i]) if raw_indices is not None and not torch.is_tensor(raw_indices)
                            else (int(raw_indices[i].item()) if raw_indices is not None else i),
                            "path": raw_paths[i] if raw_paths is not None else None,
                            "eeg_path": raw_eeg_paths[i] if raw_eeg_paths is not None else None,
                            "target": int(target[i].item()),
                            "pred": int(pred[i].item()),
                            "correct": bool(correct[i].item()),
                        }
                        records.append(rec)
                else:
                    raise ValueError(f"Unexpected preds.shape={preds.shape}")

                if self.is_main:
                    pbar.update(1)

        if self.is_main:
            pbar.close()
            records = sorted(records, key=lambda x: x["index"])

            with open(jsonl_path, "w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            with open(txt_path, "w", encoding="utf-8") as f:
                for rec in records:
                    if "joint_correct" in rec:
                        f.write(
                            f"index={rec['index']} | "
                            f"path={rec['path']} | "
                            f"target={rec['target']} | "
                            f"pred={rec['pred']} | "
                            f"val_correct={int(rec['val_correct'])} | "
                            f"aro_correct={int(rec['aro_correct'])} | "
                            f"joint_correct={int(rec['joint_correct'])}\n"
                        )
                    else:
                        f.write(
                            f"index={rec['index']} | "
                            f"path={rec['path']} | "
                            f"target={rec['target']} | "
                            f"pred={rec['pred']} | "
                            f"correct={int(rec['correct'])}\n"
                        )

        if self.is_main and len(records) > 0:
            if "joint_correct" in records[0]:
                val_acc = np.mean([r["val_correct"] for r in records])
                aro_acc = np.mean([r["aro_correct"] for r in records])
                joint_acc = np.mean([r["joint_correct"] for r in records])
                mean_acc = 0.5 * (val_acc + aro_acc)

                print(f"[{model_name}] val_acc   = {val_acc:.4f}")
                print(f"[{model_name}] aro_acc   = {aro_acc:.4f}")
                print(f"[{model_name}] mean_acc  = {mean_acc:.4f}")  # _eval의 acc와 대응
                print(f"[{model_name}] joint_acc = {joint_acc:.4f}")  # 둘 다 맞은 비율
            else:
                acc = np.mean([r["correct"] for r in records])
                print(f"[{model_name}] acc = {acc:.4f}")

    def train(self, epochs, save_every):
        self.time_stamp = self._time_stamp()
        epochs_no_improve = 0

        for epoch in range(epochs):
            self._run_epoch_single(epoch)

            metrics = self._eval()

            if dist.is_available() and dist.is_initialized():
                dist.barrier()

            should_stop = torch.tensor(0, device=self.device)

            if self.is_main:
                acc = metrics["acc"]

                if acc > self.best_metrics["acc"]:
                    self.best_metrics = metrics
                    self.best_epoch = self.current_epoch
                    epochs_no_improve = 0

                    ckpt_dir = os.path.join(self.log_dir, "checkpoints")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    best_path = os.path.join(ckpt_dir, "best.pt")

                    checkpoint = getattr(self.model, "module", self.model).state_dict()
                    torch.save(checkpoint, best_path)

                    print(
                        f"[BEST] epoch={self.current_epoch} | "
                        f"ACC={metrics['acc']:.4f}, "
                        f"F1_w={metrics['f1_weighted']:.4f}"
                    )

                    with open(self.best_log_path, "a") as f:
                        f.write(
                            f"epoch={self.current_epoch},"
                            f"acc={metrics['acc']:.6f},"
                            f"uar={metrics['uar']:.6f},"
                            f"f1_macro={metrics['f1_macro']:.6f},"
                            f"f1_weighted={metrics['f1_weighted']:.6f}\n"
                        )

                else:
                    epochs_no_improve += 1
                    if self.patience > 0 and epochs_no_improve >= self.patience:
                        print(
                            f"[Early Stop] No improvement for {self.patience} epochs "
                            f"(best epoch={self.best_epoch}). Stopping."
                        )
                        should_stop = torch.tensor(1, device=self.device)

            if dist.is_available() and dist.is_initialized():
                dist.broadcast(should_stop, src=0)

            self.current_epoch += 1

            if should_stop.item() == 1:
                break

        return self.best_metrics

    # def train(self, epochs, save_every):
    #     self.time_stamp = self._time_stamp()
    #     epochs_no_improve = 0
    #
    #     for epoch in range(epochs):
    #         self._run_epoch_single(epoch)
    #
    #         metrics = self._eval()
    #
    #         should_stop = torch.tensor(0, device=self.device)
    #
    #         if self.is_main:
    #             acc = metrics["acc"]
    #
    #             if acc > self.best_metrics["acc"]:
    #                 self.best_metrics = metrics
    #                 self.best_epoch = self.current_epoch
    #                 epochs_no_improve = 0
    #
    #                 ckpt_dir = os.path.join(self.log_dir, "checkpoints")
    #                 os.makedirs(ckpt_dir, exist_ok=True)
    #                 best_path = os.path.join(ckpt_dir, "best.pt")
    #
    #                 checkpoint = getattr(self.model, "module", self.model).state_dict()
    #                 torch.save(checkpoint, best_path)
    #
    #                 print(
    #                     f"[BEST] epoch={self.current_epoch} | "
    #                     f"ACC={metrics['acc']:.4f}, "
    #                     f"F1_w={metrics['f1_weighted']:.4f}"
    #                 )
    #
    #                 # best.txt saving
    #                 with open(self.best_log_path, "a") as f:
    #                     f.write(
    #                         f"epoch={self.current_epoch},"
    #                         f"acc={metrics['acc']:.6f},"
    #                         f"uar={metrics['uar']:.6f},"
    #                         f"f1_macro={metrics['f1_macro']:.6f},"
    #                         f"f1_weighted={metrics['f1_weighted']:.6f}\n"
    #                     )
    #
    #             else:
    #                 epochs_no_improve += 1
    #                 if self.patience > 0 and epochs_no_improve >= self.patience:
    #                     print(
    #                         f"[Early Stop] No improvement for {self.patience} epochs "
    #                         f"(best epoch={self.best_epoch}). Stopping."
    #                     )
    #                     should_stop = torch.tensor(1, device=self.device)
    #
    #         if dist.is_available() and dist.is_initialized():
    #             dist.broadcast(should_stop, src=0)
    #
    #         self.current_epoch += 1
    #
    #         if should_stop.item() == 1:
    #             break
    #
    #     return self.best_metrics

    def test_only(self, save_dir, model_name="model"):
        # self.time_stamp = self._time_stamp()
        # metrics = self._eval()
        #
        # ckpt_dir = os.path.join(self.log_dir, "sample")
        # os.makedirs(ckpt_dir, exist_ok=True)
        # sample_path = os.path.join(ckpt_dir, "sample.txt")

        self.time_stamp = self._time_stamp()
        pred_dir = os.path.join(save_dir, "sample_predictions")
        os.makedirs(pred_dir, exist_ok=True)
        self.save_test_predictions(model_name=model_name, save_dir=pred_dir)
