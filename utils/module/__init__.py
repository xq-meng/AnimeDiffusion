import torch.nn as nn
import copy


def copy_module(module: nn.Module, requires_grad=None):
    if not module:
        return None
    new_module = copy.deepcopy(module)
    if requires_grad is not None:
        for parameter in new_module.parameters():
            parameter.requires_grad = requires_grad
    return new_module


def zero_module(module: nn.Module):
    if not module:
        return None
    for parameter in module.parameters():
        parameter.detach().zero_()
    return module