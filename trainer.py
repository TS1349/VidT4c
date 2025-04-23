import idr_torch

import math
import time
import torch
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel as DDP

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
                 ):

        self.time_stamp = "00000000" 
        self.experiment_name = experiment_name
        self.checkpoint_dir = checkpoint_dir

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

    def _save_checkpoint(self):
        checkpoint = self.model.module.state_dict()
        checkpoint_path =\
            f"{self.checkpoint_dir}/{self.time_stamp}_{self.experiment_name}_{self.current_epoch}.pt"

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
        nn.utils.clip_grad_norm_(parameters = self.model.parameters(),
                                 max_norm = 1,
                                 )

        self.optimizer.step()
        self.lr_scheduler.step()
        print(f"ST, GPU ID : {self.gpu_id}, : {self.step}, lr : {self.optimizer.param_groups[0]['lr']}\
")
        self.step += 1
        return loss.item()

    def _time_stamp(self):
        return str(math.floor(time.time()))


    def _run_epoch(self, epoch) -> None:
        self.model.train()
        self.training_dataloader.sampler.set_epoch(epoch)  # multi-gpu
        avg_loss = 0
        bn = 0
        for batch_number, sample in enumerate(self.training_dataloader):
            for k in sample:
                sample[k] = sample[k].to(self.gpu, non_blocking=True)
            avg_loss += self._run_batch(sample)
            bn = batch_number
        avg_loss /= (bn + 1)
        print(f"TR, GPU ID : {self.gpu_id}, EP: {epoch}, \
loss : {avg_loss}, lr : {self.optimizer.param_groups[0]['lr']}\
")

            #DEBUG
            #if batch_number > 2:
            #    break

    def _eval(self) -> None:
        self.model.eval()
        size = len(self.validation_dataloader.dataset)
        num_batches = len(self.validation_dataloader)
        test_loss, correct = 0, 0

        with torch.inference_mode():
            counter = 0
            for sample in self.validation_dataloader:
                self.training_dataloader.sampler.set_epoch(counter)  # multi-gpu
                for k in sample:
                    sample[k] = sample[k].to(self.gpu, non_blocking=True)
                predictions = self.model(sample)
                test_loss += self.loss_function(predictions, sample["output"]).item()
                correct +=\
                    (predictions.argmax(1) == sample["output"]).type(torch.float).sum(0)
                counter += 1
            #DEBUG
                #if counter > 2:
                #    break

            test_loss /= num_batches
            correct /= size
            if self.gpu_id == 0:
                print(f"\
VA, GPU ID : {self.gpu_id}, EP: {self.current_epoch},\
loss : {test_loss}, accuracy : {correct.tolist()}")

    def train(self, epochs, save_every):
        self.time_stamp = self._time_stamp()
        for epoch in range(epochs):
            self._run_epoch(epoch)
            if idr_torch.rank ==  0:
                self._eval()
                if ((self.current_epoch + 1) % save_every == 0):
                    self._save_checkpoint()
            self.current_epoch += 1

