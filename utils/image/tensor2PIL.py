import torch
from torchvision.transforms import ToPILImage


def tensor2PIL(tsr: torch.Tensor):
    assert len(tsr.shape) == 4
    tsr = (torch.clamp(tsr.detach().cpu(), min=-1, max=1).float() + 1.) / 2.
    batch_size = tsr.shape[0]
    ret = []
    for c in range(batch_size):
        ret.append(ToPILImage()(tsr[c]))
    return ret