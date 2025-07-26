import math
import time
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import os

try:
    import tqdm
except ImportError:
    tqdm = None

class PTrainer:
    def __init__(self,
                 model,
                 optimizer,
                 lr_scheduler,
                 loss_function,
                 training_dataloader,
                 validation_dataloader,
                 gpu_id,
                 gpu,
                 checkpoint_dir,
                 experiment_name="",
                 use_tqdm = True,
                 ):

        self.time_stamp = "00000000"
        self.experiment_name = experiment_name
        self.checkpoint_dir = checkpoint_dir

        # For saving the validation loss and accuracy
        self.log_dir = os.path.join(self.checkpoint_dir, self.experiment_name)
        self.checkpoint_subdir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(self.checkpoint_subdir, exist_ok=True)

        self.val_log_path = os.path.join(self.log_dir, "val_loss.log")

        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader

        self.loss_function = loss_function
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # multi-gpu
        self.gpu_id = gpu_id
        self.gpu = gpu
        self.model = DDP(model, device_ids=[self.gpu_id])
        self.step = 0
        self.current_epoch = 0
        self.use_tqdm = use_tqdm

    def _save_checkpoint(self):
        checkpoint = self.model.module.state_dict()

        checkpoint_path =\
            f"{self.checkpoint_subdir}/{self.time_stamp}_{self.experiment_name}_{self.current_epoch}.pt"

        torch.save(
            obj=checkpoint,
            f=checkpoint_path,
        )
        print(f"Epoch {self.current_epoch}: checkpoint saved at {checkpoint_path}")

    def _run_batch(self, sample):
        self.optimizer.zero_grad()
        output = self.model(sample)
        loss = self.loss_function(output, sample["output"])
        loss.backward()

        self.optimizer.step()
        self.lr_scheduler.step()

        self.step += 1
        return loss.item()

    def _time_stamp(self):
        return str(math.floor(time.time()))

    def _run_epoch(self, epoch):
        self.model.train()
        self.training_dataloader.sampler.set_epoch(epoch)  # multi-gpu

        avg_loss = 0

        # Replace printing accuracy at each step to pbar based on epochs
        if (tqdm is not None) and self.use_tqdm:
            pbar = tqdm.tqdm(total=len(self.training_dataloader), desc=f"Train Epoch {epoch}")
        else:
            pbar = None

        for batch_number, sample in enumerate(self.training_dataloader):
            for k in sample:
                sample[k] = sample[k].to(self.gpu, non_blocking=True)
            batch_loss = self._run_batch(sample)
            avg_loss += batch_loss

            running_avg_loss = avg_loss / (batch_number + 1)

            if (pbar is not None):
                pbar.set_postfix({
                    # 'batch_loss': f'{batch_loss:.4f}',
                    'avg_loss': f'{running_avg_loss:.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
                })
                pbar.update(1)
            else:
                print(f"T/G[{self.gpu_id}]/E[{epoch}]/B[{batch_number}] : L={running_avg_loss:.4f}; lr={self.optimizer.param_group[0]["lr"]:.6f}")

        if (pbar is not None):
            pbar.close()


    def _eval(self) -> None:
        self.model.eval()
        num_batches = len(self.validation_dataloader)
        total_loss = 0.0
        total_samples = 0

        total_correct = None
        if (tqdm is not None) and self.use_tqdm:
            pbar = tqdm.tqdm(total=num_batches, desc=f"Validate Epoch {self.current_epoch}")
        else:
            pbar = None

        # Temo : had to add these so that pyright would shut up
        # <error_sentinels>
        pred = None
        global_loss = -1.0
        val_acc = -1.0
        aro_acc = -1.0
        dom_acc = -1.0
        avg_acc = -1.0
        # </error_sentinels>

        with torch.inference_mode():
            for _, sample in enumerate(self.validation_dataloader):
                batch_size_now = sample["output"].size(0)
                total_samples += batch_size_now

                for k in sample:
                    sample[k] = sample[k].to(self.gpu, non_blocking=True)

                predictions = self.model(sample)
                batch_loss_value = self.loss_function(predictions, sample["output"]).item()
                total_loss += batch_loss_value * batch_size_now
                global_loss = total_loss / total_samples

                pred = predictions.argmax(1)  # [batch_size, N]
                target = sample["output"]     # same

                # Devide accuracy at each emotion anntation
                if pred.dim() == 2 and pred.size(1) == 3:
                    # [valence, arousal, dominance / motivation] -> MDMER, Emognition
                    if total_correct is None:
                        total_correct = {"valence": 0, "arousal": 0, "dominance": 0}

                    val_correct = (pred[:, 0] == target[:, 0]).sum().item()
                    aro_correct = (pred[:, 1] == target[:, 1]).sum().item()
                    dom_correct = (pred[:, 2] == target[:, 2]).sum().item()

                    total_correct["valence"] += val_correct
                    total_correct["arousal"] += aro_correct
                    total_correct["dominance"] += dom_correct

                    val_acc = total_correct["valence"] / total_samples
                    aro_acc = total_correct["arousal"] / total_samples
                    dom_acc = total_correct["dominance"] / total_samples

                    acc_str = f"Val: {val_acc:.3f} | Aro: {aro_acc:.3f} | Dom: {dom_acc:.3f}"

                elif pred.dim() == 1:
                    # Category classification -> EAV
                    if total_correct is None:
                        total_correct = 0
                    total_correct += (pred == target).sum().item()
                    avg_acc = total_correct / total_samples
                    acc_str = f"Cat: {avg_acc:.3f}"

                else:
                    raise ValueError(f"Unexpected prediction shape: {pred.shape}")

                if (pbar is not None):
                    pbar.set_postfix_str(f"val_loss={global_loss:.4f}, {acc_str}")
                    pbar.update(1)
                else:
                    print(f"V/G[{self.gpu_id}]/val_loss={global_loss:.4f}, {acc_str}")

        if (pbar is not None):
            pbar.close()

        # Save the validation logger in the checkpoints dir
        with open(self.val_log_path, "a") as f:
            if (pred is not None):
                if pred.dim() == 2 and pred.size(1) == 3:
                    f.write(f"{self.current_epoch},{global_loss:.6f},{val_acc:.6f},{aro_acc:.6f},{dom_acc:.6f}\n")
                else:
                    f.write(f"{self.current_epoch},{global_loss:.6f},{avg_acc:.6f}\n")
            else:
                f.write(f"Error: pred is None")


    def train(self, epochs, save_every):
        self.time_stamp = self._time_stamp()
        for epoch in range(epochs):
            self._run_epoch(epoch)
            # Validation only uses single GPU
            if self.gpu_id == 0:
                self._eval()
                if ((self.current_epoch + 1) % save_every == 0):
                    self._save_checkpoint()
            self.current_epoch += 1
