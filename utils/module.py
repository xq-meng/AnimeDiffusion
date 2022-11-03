from abc import abstractmethod
import torch.nn as nn


class Module(nn.Module):
    
    @abstractmethod
    def forward(self, x, t):
        raise NotImplemented

        
class Sequential(nn.Sequential, Module):
    
    @abstractmethod
    def forward(self, x, t):
        for layer in self:
            x = layer(x, t) if isinstance(layer, Module) else layer(x)
        return x