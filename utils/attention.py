import torch
import torch.nn as nn
import math


class QKVAttention(nn.Module):
    """
    Reference: https://github.com/LouisRouss/Diffusion-Based-Model-for-Colorization/blob/main/src/network.py
    """

    def __init__(
        self,
        num_heads
    ):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, qkv):
        batch_size, width, length = qkv.shape
        assert width % (3 * self.num_heads) == 0
        channels = width // (3 * self.num_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(channels))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(batch_size * self.num_heads, channels, length),
            (k * scale).view(batch_size * self.num_heads, channels, length),
        )
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum(
            "bts,bcs->bct",
            weight,
            v.reshape(batch_size * self.num_heads, channels, length)
        )
        return a.reshape(batch_size, -1, length)