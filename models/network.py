import torch
import torch.nn as nn
from unet import UNet


class Network(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()


if __name__ == '__main__':
    b, c, h = 3, 6, 64
    model = UNet(image_size=h, channel_in=c)
    x = torch.randn((b, c, h, h))
    h = model(x)
    print(h.size())