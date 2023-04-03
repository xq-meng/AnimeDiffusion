import torch
import torch.nn as nn
from typing import Union


class DistributedParallel(nn.parallel.DistributedDataParallel):
    def __getattr__(self, name: str) -> Union[torch.Tensor, 'nn.Module']:
        # basic __getattr__
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        # advanced __getattr__
        if hasattr(self, 'module') and hasattr(getattr(self, 'module'), name):
            return getattr(getattr(self, 'module'), name)
        # raise ex.
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def state_dict(self):
        if hasattr(self, 'module'):
            return self.module.state_dict()
        return super().state_dict()

    def load_state_dict(self, state_dict, strict: bool = True):
        if hasattr(self, 'module'):
            return self.module.load_state_dict(state_dict=state_dict, strict=strict)
        return super().load_state_dict(state_dict, strict)