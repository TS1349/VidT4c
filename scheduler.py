from torch.optim.lr_scheduler import LambdaLR
import math

#from https://github.com/jeonsworld/ViT-pytorch/blob/main/utils/scheduler.py

class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self,
                 optimizer,
                 warmup_steps,
                 t_total,
                 cycles=.5,
                 last_epoch=-1,
                 ):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(
            optimizer,
            self.lr_lambda,
            last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))


class SteppedScheduler(LambdaLR):
    def __init__(self,
                 optimizer,
                 epochs = [11, 14],
                 lrs = [1, 0.1, 0.01],
                 last_epoch = -1,
                 ):
        self.epoch_idx = 0
        self.epochs = epochs
        self.lrs = lrs
        super(SteppedScheduler, self).__init__(
            optimizer,
            self.lr_lambda,
            last_epoch=last_epoch
        )


    def lr_lambda(self,step):
        if step >= self.epochs[self.epoch_idx]:
            self.epoch_idx +=1
        return self.lrs[self.epoch_idx]

