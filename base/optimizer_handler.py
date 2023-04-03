import math
import torch
import torch.optim
import utils.pythonic


class OptimizerHandler(object):
    def __init__(
        self,
        name,
        parameters,
        lr,
        min_lr=0.,
        lr_decay=True,
        warmup_epochs=-1,
        **kwargs,
    ):
        # optimizer initialization
        optim_class = getattr(torch.optim, name)
        optim_option = utils.pythonic.argument_autofilling(optim_class.__init__, kwargs=kwargs)
        optim_option['lr'] = lr
        self.optimizer = optim_class(params=parameters, **optim_option)
        # parameters for optimizer scheduler
        self.warmup_epochs = warmup_epochs
        self.base_lr = lr
        self.min_lr = min_lr
        self.lr_decay = lr_decay

    def update_lr(self, epoch, target_epoch):
        if epoch < self.warmup_epochs:
            lr = self.min_lr + (self.base_lr - self.min_lr) * epoch / self.warmup_epochs
        elif self.lr_decay:
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1. + math.cos(math.pi * (epoch - self.warmup_epochs) / (target_epoch - self.warmup_epochs)))
        else:
            lr = self.base_lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr * param_group["lr_scale"] if "lr_scale" in param_group else lr
        return lr

    def learning_rate(self):
        return self.optimizer.param_groups[0]["lr"]

    def load_checkpoint(self, path_to_checkpoint):
        ckpt = torch.load(path_to_checkpoint)
        self.optimizer.load_state_dict(ckpt['optimizer_dict'])

    def save_checkpoint(self, path_to_checkpoint):
        torch.save({'optim_dict': self.optimizer.state_dict()}, path_to_checkpoint)