import math
import time
import torch
import torch.nn.functional as F
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
                 use_tqdm = False,
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

        self.model = model
        for param in self.model.parameters():
            param.requires_grad = True

        self.loss_function = loss_function
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # multi-gpu
        self.gpu_id = gpu_id
        self.gpu = gpu
        self.step = 0
        self.current_epoch = 0
        self.use_tqdm = use_tqdm

    def _save_checkpoint(self):
        checkpoint = self.model.state_dict()

        checkpoint_path =\
            f"{self.checkpoint_subdir}/{self.time_stamp}_{self.experiment_name}_{self.current_epoch}.pt"

        torch.save(
            obj=checkpoint,
            f=checkpoint_path,
        )
        print(f"Epoch {self.current_epoch}: checkpoint saved at {checkpoint_path}")

    def _time_stamp(self):
        return str(math.floor(time.time()))

    def _run_epoch_single(self, epoch) -> None:
        torch.set_grad_enabled(True)
        self.model.train()

        self.training_dataloader.sampler.set_epoch(epoch)

        total_loss = 0
        if (tqdm is not None) and self.use_tqdm:
                pbar = tqdm.tqdm(total=len(self.training_dataloader), desc=f"Train Epoch {epoch}")
        else:
            pbar = None

        for batch_number, sample in enumerate(self.training_dataloader):
            # Move sample to device
            sample = {key: val.cuda(self.gpu) for key, val in sample.items()
                    if val is not None}

            self.optimizer.zero_grad()
            output = self.model(sample)
            target = sample["output"]
            
            # # loss = self.loss_function(output.float(), target)
            v_loss = self.loss_function(output[:, :, 0].float(), target[:, 0])
            a_loss = self.loss_function(output[:, :, 1].float(), target[:, 1])
            loss = 0.5 * (v_loss + a_loss)

            # Check actual prediction class for each sample
            # print("Pred:", output.argmax(1), "Target:",target)

            # Backward
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            # # Optional: check gradients
            # for name, param in self.model.named_parameters():
            #     if param.grad is not None:
            #         print(f"[Grad] {name}: {param.grad.sum().item():.4f}")
            #     else:
            #         print(f"[Grad] {name}: None")

            total_loss += loss.item()
            avg_loss = total_loss / (batch_number + 1)
            if (pbar is not None):
                pbar.set_postfix(avg_loss=f'{avg_loss:.4f}',
                                 lr=f'{self.optimizer.param_groups[0]["lr"]:.6f}')
                pbar.update(1)
            else:
                print(f"T/G[{self.gpu_id}]/E[{epoch}]/B[{batch_number}] : L={avg_loss:.4f}; lr={self.optimizer.param_groups[0]["lr"]:.6f}")



    def _eval(self,epoch) -> None:
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
        avg_acc = -1.0
        # </error_sentinels>

        with torch.inference_mode():
            for batch_number, sample in enumerate(self.validation_dataloader):
                batch_size_now = sample["output"].size(0)
                total_samples += batch_size_now

                sample = {key: val.cuda(self.gpu) for key, val in sample.items()
                    if val is not None}

                target = sample["output"]
                predictions = self.model(sample)
                
                v_loss = self.loss_function(predictions[:, :, 0].float(), target[:, 0])
                a_loss = self.loss_function(predictions[:, :, 1].float(), target[:, 1])
                batch_loss_value = 0.5 * (v_loss + a_loss)

                total_loss += batch_loss_value * batch_size_now
                global_loss = total_loss / total_samples

                pred = predictions.argmax(1)  # [batch_size, N]
                # print(pred, target)

                # Devide accuracy at each emotion anntation
                if pred.dim() == 2 and pred.size(1) == 2:
                    # [valence, arousal, dominance / motivation] -> MDMER, Emognition
                    if total_correct is None:
                        total_correct = {"valence": 0, "arousal": 0}

                    val_correct = (pred[:, 0] == target[:, 0]).sum().item()
                    aro_correct = (pred[:, 1] == target[:, 1]).sum().item()

                    # val_correct = (v_expect == target[:, 0]).sum().item()
                    # aro_correct = (a_expect == target[:, 1]).sum().item()

                    total_correct["valence"] += val_correct
                    total_correct["arousal"] += aro_correct

                    val_acc = total_correct["valence"] / total_samples
                    aro_acc = total_correct["arousal"] / total_samples

                    acc_str = f"Val: {val_acc:.3f} | Aro: {aro_acc:.3f}"

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
                    print(f"V/G[{self.gpu_id}]/E[{epoch}]/B[{batch_number}] : L={batch_loss_value:.4f}; acc={acc_str}")

        if (pbar is not None):
            pbar.close()

        # Save the validation logger in the checkpoints dir
        with open(self.val_log_path, "a") as f:
            if (pred is not None):
                if pred.dim() == 2 and pred.size(1) == 2:
                    f.write(f"{self.current_epoch},{global_loss:.6f},{val_acc:.6f},{aro_acc:.6f}\n")
                else:
                    f.write(f"{self.current_epoch},{global_loss:.6f},{avg_acc:.6f}\n")
            else:
                f.write(f"Error: pred is None")


    def train(self, epochs, save_every):
        self.time_stamp = self._time_stamp()

        for epoch in range(epochs):
            self._run_epoch_single(epoch)
            if (0 == self.gpu_id):
                self._eval(epoch)
                if ((self.current_epoch + 1) % save_every == 0):
                    self._save_checkpoint()
            self.current_epoch += 1
