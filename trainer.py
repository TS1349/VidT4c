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
        fusion,
        weight_ce,
        training_dataloader,
        validation_dataloader,
        gpu_id,
        gpu,  # kept for backward-compat; not used
        checkpoint_dir,
        experiment_name="",
        patience=20,
        per_clip_aux_loss=0.0,
        grad_clip=0.0,
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

        self.loss_function = loss_function
        self.ce_val = loss_function_v
        self.ce_aro = loss_function_a
        self.CE = torch.nn.CrossEntropyLoss(label_smoothing=0.1)

        self.fusion = fusion

        self.w_ce = weight_ce

        self.patience = patience

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

        self.per_clip_aux_loss_w = per_clip_aux_loss
        self.grad_clip = float(grad_clip)

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
            sample = {
                k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in sample.items()
                if v is not None
            }

            self.optimizer.zero_grad()

            per_clip_logits = None

            model_out = self.model(sample)
            # (output, per_clip_logits) in the K-clip per_clip_aux_loss path, else output.
            if isinstance(model_out, tuple):
                output, per_clip_logits = model_out
            else:
                output = model_out
                per_clip_logits = None
            coral_loss = torch.zeros((), device=self.device)

            target = sample["output"]

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

            # GCNII decorrelation aux (stashed on the model during forward).
            _inner = self.model.module if hasattr(self.model, 'module') else self.model
            _faux = getattr(getattr(_inner, 'gcn_region', None), '_fusion_aux', None)
            if _faux is not None:
                combined = combined + _faux

            # --deep_fuse: auxiliary supervision on each stage logit (video branch,
            # eeg branch, fused). Branches are stashed as [B, 2C]; split into V/A.
            _dfb = getattr(_inner, '_deep_fuse_branches', None)
            if _dfb is not None and output.dim() == 3:
                _dfw = float(getattr(_inner, 'deep_fuse_w', 0.5))
                _C = output.size(1)
                _df_aux = 0.0
                for _bl in _dfb:
                    _df_aux = _df_aux + 0.5 * (
                        self.loss_function(_bl[:, :_C].float(), target[:, 0])
                        + self.loss_function(_bl[:, _C:].float(), target[:, 1]))
                combined = combined + _dfw * _df_aux

            loss = combined
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               max_norm=self.grad_clip)
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

                sample = {
                    k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v)
                    for k, v in sample.items()
                    if v is not None
                }
                target = sample["output"]
                bs = target.size(0)

                model_out = self.model(sample)
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

            # Branch expert metrics
            eeg_val_acc = (total_correct_eeg_val / denom).item()
            eeg_aro_acc = (total_correct_eeg_aro / denom).item()
            video_val_acc = (total_correct_video_val / denom).item()
            video_aro_acc = (total_correct_video_aro / denom).item()
            gcn_val_acc = (total_correct_gcn_val / denom).item()
            gcn_aro_acc = (total_correct_gcn_aro / denom).item()

            eeg_mean_acc = 0.5 * (eeg_val_acc + eeg_aro_acc)
            video_mean_acc = 0.5 * (video_val_acc + video_aro_acc)
            gcn_mean_acc = 0.5 * (gcn_val_acc + gcn_aro_acc)

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

        # --dump_affinity: capture per-sample GCN affinity, node-type degrees,
        # cross-block coupling and region_alpha. Node layout: [0:K_v) video clips,
        # [K_v:K_v+K_e) EEG clips, [K_v+K_e:N) regions.
        _affin = getattr(self, 'dump_affinity', False)
        _cap = {}
        _hooks = []
        _affin_rows = []
        if _affin:
            _inner = self.model.module if hasattr(self.model, 'module') else self.model
            _greg = getattr(_inner, 'gcn_region', None)

            def _pre_gcn1(mod, args):
                if len(args) >= 2 and torch.is_tensor(args[1]):
                    _cap['adj'] = args[1].detach()

            def _pre_region(mod, args, kwargs):
                _cap['K_v'] = int(kwargs.get('num_video_nodes', 1))
                _cap['K_e'] = int(kwargs.get('num_eeg_nodes', 1))

            def _post_alpha(mod, args, out):
                if torch.is_tensor(out):
                    _cap['alpha'] = torch.sigmoid(out.detach()).squeeze(-1)  # [B, R]

            # GCNII bypasses gcn1, stashing adj/H on the module instead.
            def _post_region(mod, args, kwargs, out):
                if hasattr(mod, '_dump_adj'):
                    _cap['adj'] = mod._dump_adj
                if hasattr(mod, '_dump_H'):
                    _cap['H'] = mod._dump_H
                if hasattr(mod, '_dump_Hpre'):
                    _cap['Hpre'] = mod._dump_Hpre

            if _greg is not None and hasattr(_greg, 'gcn1'):
                _hooks.append(_greg.gcn1.register_forward_pre_hook(_pre_gcn1))
                _hooks.append(_greg.register_forward_pre_hook(_pre_region, with_kwargs=True))
                _hooks.append(_greg.register_forward_hook(_post_region, with_kwargs=True))
                if hasattr(_inner, 'region_gate_mlp'):
                    _hooks.append(_inner.region_gate_mlp.register_forward_hook(_post_alpha))
                _cap['gate_w'] = (float(torch.sigmoid(_greg.weight.detach()).mean().item())
                                  if hasattr(_greg, 'weight') else None)
                _cap['ii_gamma'] = (float(_greg.ii_gamma.detach().item())
                                    if hasattr(_greg, 'ii_gamma') else None)

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

                    # Softmax prob of the true class on each axis (V/A).
                    bs = target.size(0)
                    probs = torch.softmax(preds.float(), dim=1)  # [B, C, 2]
                    ar = torch.arange(bs, device=preds.device)
                    prob_true_v = probs[ar, target[:, 0], 0]
                    prob_true_a = probs[ar, target[:, 1], 1]

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
                            "prob_true_v": float(prob_true_v[i].item()),
                            "prob_true_a": float(prob_true_a[i].item()),
                        }
                        records.append(rec)

                        if _affin and 'adj' in _cap:
                            adj = _cap['adj']
                            if i < adj.size(0):
                                Kv, Ke = _cap.get('K_v', 1), _cap.get('K_e', 1)
                                N = adj.size(1); R = N - Kv - Ke
                                a = adj[i]                       # [N, N] affinity
                                vs, es, rs = slice(0, Kv), slice(Kv, Kv + Ke), slice(Kv + Ke, N)

                                def _blk(rr, cc):
                                    sub = a[rr, cc]
                                    return float(sub.mean().item()) if sub.numel() else 0.0

                                arow = {
                                    "index": rec["index"],
                                    "val_correct": rec["val_correct"],
                                    "aro_correct": rec["aro_correct"],
                                    "joint_correct": rec["joint_correct"],
                                    "Kv": Kv, "Ke": Ke, "R": R,
                                    # node-type degree = mean row-sum of A over that block
                                    "deg_video": float(a[vs, :].sum(-1).mean().item()),
                                    "deg_eeg": float(a[es, :].sum(-1).mean().item()),
                                    "deg_region": (float(a[rs, :].sum(-1).mean().item()) if R > 0 else 0.0),
                                    # cross-block mean affinity (modality coupling)
                                    "vv": _blk(vs, vs), "ee": _blk(es, es),
                                    "rr": (_blk(rs, rs) if R > 0 else 0.0),
                                    "ve": _blk(vs, es),
                                    "vr": (_blk(vs, rs) if R > 0 else 0.0),
                                    "er": (_blk(es, rs) if R > 0 else 0.0),
                                }
                                if R > 0:
                                    # per-region breakdown: each region node's total degree,
                                    # and its mean affinity to the video block and the EEG node.
                                    arow["region_deg"] = a[rs, :].sum(-1).detach().cpu().tolist()
                                    arow["region_to_video"] = a[rs, vs].mean(-1).detach().cpu().tolist()
                                    arow["region_to_eeg"] = a[rs, es].mean(-1).detach().cpu().tolist()
                                if 'alpha' in _cap and _cap['alpha'].dim() >= 2 and i < _cap['alpha'].size(0):
                                    arow["region_alpha"] = _cap['alpha'][i].detach().cpu().tolist()
                                if _cap.get('ii_gamma') is not None:
                                    arow["ii_gamma"] = _cap['ii_gamma']
                                # Oversmoothing (all-node sim ~1) + cross-modal feature
                                # collapse (video-mean vs eeg-mean sim ~1). Computed on the
                                # graph-output features (H) and, for the base path, also on
                                # the pre-GCN inputs (Hpre) to see the collapse happen.
                                def _feat_stats(feat_i):
                                    Hn = torch.nn.functional.normalize(feat_i, dim=-1)
                                    Nn = feat_i.size(0)
                                    sim = Hn @ Hn.t()
                                    off = (sim.sum() - Nn) / max(1, Nn * Nn - Nn)
                                    ve = torch.nn.functional.cosine_similarity(
                                        Hn[vs].mean(0).unsqueeze(0), Hn[es].mean(0).unsqueeze(0))
                                    return float(off.item()), float(ve.item())
                                if 'H' in _cap and i < _cap['H'].size(0):
                                    arow["H_node_sim"], arow["H_ve_sim"] = _feat_stats(_cap['H'][i])
                                if 'Hpre' in _cap and i < _cap['Hpre'].size(0):
                                    arow["Hpre_node_sim"], arow["Hpre_ve_sim"] = _feat_stats(_cap['Hpre'][i])
                                _affin_rows.append(arow)

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

        for _h in _hooks:
            _h.remove()

        if self.is_main:
            pbar.close()
            records = sorted(records, key=lambda x: x["index"])

            with open(jsonl_path, "w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if _affin and _affin_rows:
                _affin_rows = sorted(_affin_rows, key=lambda x: x["index"])
                affin_path = os.path.join(save_dir, f"{model_name}_affinity.jsonl")
                with open(affin_path, "w", encoding="utf-8") as f:
                    if _cap.get('gate_w') is not None:
                        f.write(json.dumps({"_meta": True, "gate_w_video_mean": _cap['gate_w']}) + "\n")
                    for arow in _affin_rows:
                        f.write(json.dumps(arow, ensure_ascii=False) + "\n")
                print(f"[dump_affinity] wrote {len(_affin_rows)} rows -> {affin_path} "
                      f"(gate_w_video_mean={_cap.get('gate_w')})")

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
        import gc
        try:
            import psutil
            _proc = psutil.Process()
        except ImportError:
            _proc = None

        self.time_stamp = self._time_stamp()
        epochs_no_improve = 0

        for epoch in range(epochs):
            _inner = self.model.module if hasattr(self.model, 'module') else self.model
            if hasattr(_inner, 'current_epoch'):
                _inner.current_epoch = epoch
            self._run_epoch_single(epoch)

            metrics = self._eval()

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if self.is_main and _proc is not None:
                _rss_gb = _proc.memory_info().rss / (1024 ** 3)
                print(f'[mem] epoch {epoch} end: rank0 RSS = {_rss_gb:.2f} GB')

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
                    # Atomic write (tmp + rename); tolerate disk-full without crashing.
                    _tmp_path = best_path + f'.tmp.{os.getpid()}'
                    try:
                        torch.save(checkpoint, _tmp_path)
                        os.replace(_tmp_path, best_path)
                        _saved = True
                    except Exception as _e:
                        _saved = False
                        if os.path.exists(_tmp_path):
                            try:
                                os.unlink(_tmp_path)
                            except OSError:
                                pass
                        print(f'[BEST][WARN] checkpoint save failed: {_e}. '
                              f'Continuing without overwriting best.pt.')

                    print(
                        f"[BEST] epoch={self.current_epoch} | "
                        f"ACC={metrics['acc']:.4f}, "
                        f"F1_w={metrics['f1_weighted']:.4f}"
                        + ('' if _saved else ' (ckpt NOT saved — disk issue)')
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

    def test_only(self, save_dir, model_name="model"):
        self.time_stamp = self._time_stamp()
        pred_dir = os.path.join(save_dir, "sample_predictions")
        os.makedirs(pred_dir, exist_ok=True)
        self.save_test_predictions(model_name=model_name, save_dir=pred_dir)
