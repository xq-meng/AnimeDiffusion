import torch
import torch.nn as nn
import torch.nn.parallel
import utils.pythonic
import models
from .distributed_parallel import DistributedParallel


class ModelHandler(object):
    def __init__(
        self,
        name,
        local_rank=0,
        global_rank=1,
        checkpoint='',
        init_function='__init__',
        call_function='__call__',
        distributed_parallel=True,
        **kwargs,
    ):
        # local rank
        self.local_rank = local_rank if local_rank >= 0 and torch.cuda.is_available() else -1
        self.global_rank = global_rank if self.local_rank >= 0 else 0
        target_device = f'cuda:{self.local_rank}' if self.local_rank >= 0 else 'cpu'
        # local variables
        self.call_function = call_function
        self.distributed_parallel = distributed_parallel
        # build up model
        self.init_model(name, init_function, kwargs)
        # model to device
        self.to(target_device)
        # distribute
        if self.distributed_parallel:
            self.make_distributed()
        # checkpoint
        if checkpoint:
            self.load_checkpoint(checkpoint)

    def __call__(self, non_blocking=True, *args, **kwargs):
        # TODO: replace 'cuda()' with 'to()'.
        if self.local_rank >= 0:
            args = [x.cuda(self.local_rank, non_blocking=non_blocking) if isinstance(x, torch.Tensor) else x for x in args]
            kwargs = {k: v if not isinstance(v, torch.Tensor) else v.cuda(self.local_rank, non_blocking=non_blocking) for k, v in kwargs.items()}
        return getattr(self.model, self.call_function)(*args, **kwargs)

    def __getattr__(self, name: str):
        if self.model is not None:
            return getattr(self.model, name)

    def init_model(self, name, init_function, kwargs):
        model_class = getattr(models, name)
        model_option =  kwargs #utils.pythonic.argument_autofilling(getattr(model_class, init_function), kwargs=kwargs)
        if init_function != '__init__':
            self.model = getattr(model_class, init_function)(**model_option)
        else:
            self.model = model_class(**model_option)

    def to(self, device):
        self.device = device
        # move models to specified device.
        if hasattr(self.model, 'to'):
            self.model.to(device)
            return
        for _k in self.model.__dict__:
            if not _k.startswith('_') and isinstance(getattr(self.model, _k), (nn.Module, DistributedParallel)):
                getattr(self.model, _k).to(device)

    def get(self, attr_name):
        attr_base = self.model
        if isinstance(self.model, DistributedParallel):
            attr_base = self.model.module
        return getattr(attr_base, attr_name)

    def train(self, mode=True):
        if isinstance(self.model, (nn.Module, DistributedParallel)):
            self.model.train(mode=mode)

    def save_checkpoint(self, path_to_checkpoint):
        if self.local_rank > 0:
            return
        if isinstance(self.model, (nn.Module, DistributedParallel)):
            torch.save(self.model.state_dict(), path_to_checkpoint)
        else:
            # try to call "save_checkpoint" of your own model class.
            # if "save_checkpoint" is not implemented, raise exception.
            getattr(self.model, "save_checkpoint")(path_to_checkpoint)

    def load_checkpoint(self, path_to_checkpoint):
        ckpt = torch.load(path_to_checkpoint)
        if isinstance(self.model, (nn.Module, DistributedParallel)):
            self.model.load_state_dict(ckpt)
        else:
            # try to call "load_checkpoint" of your own model class.
            # if "load_checkpoint" is not implemented, raise exception.
            getattr(self.model, "load_checkpoint")(path_to_checkpoint)

    def make_distributed(self):
        if self.local_rank < 0:
            return
        if isinstance(self.model, DistributedParallel):
            return
        if isinstance(self.model, nn.Module):
            self.model = DistributedParallel(self.model, device_ids=[self.local_rank])
            return
        for _k in self.model.__dict__:
            if not _k.startswith('_') and isinstance(getattr(self.model, _k), nn.Module):
                if any([p.requires_grad for p in getattr(self.model, _k).parameters()]):
                    setattr(self.model, _k, DistributedParallel(getattr(self.model, _k), device_ids=[self.local_rank]))