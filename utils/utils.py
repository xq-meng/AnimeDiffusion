import torch
import math


def time_embedding(time_step, dimension, max_period=1000):
    '''
    :param[in]  time_step   torch.Tensor [N], one per batch element.
    :param[in]  dimension   the dimension of output embedding.
    :param[in]  max_period  Sinusodial position embedding. 'Attention is all you need'

    :return     torch.Tensor    [N x dimension]
    '''
    half = dimension // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=time_step.device)
    args = time_step[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dimension % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.to(t.device).gather(0, t).float()
    out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    return out


def rgb2xyz(r, g, b):
    '''
    :param[in]  r, g, b int [0, 255]
    '''
    def gamma(t):
        return math.pow((t + 0.055) / 1.055, 2.4) if t > 0.04045 else t / 12.92

    rr = gamma(r / 255.0)
    gg = gamma(g / 255.0)
    bb = gamma(b / 255.0)
    x = 0.4124564 * rr + 0.3575761 * gg + 0.1804375 * bb
    y = 0.2126729 * rr + 0.7151522 * gg + 0.0721750 * bb
    z = 0.0193339 * rr + 0.1191920 * gg + 0.9503041 * bb

    return x, y, z


def xyz2lab(x, y, z):
    def theta(t):
        return math.pow(t, 0.3333333) if t > 0.008856 else  7.787 * t + 0.1379310

    x_n = 0.950456
    y_n = 1.000000
    z_n = 1.088754
    x /= x_n
    y /= y_n
    z /= z_n
    fx = theta(x)
    fy = theta(y)
    fz = theta(z)
    L = max(116 * fy - 16.0, 0.0)
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)

    return L, a, b


def rgb2lab(r: int, g: int, b: int):
    '''
    :param[in]  r, g, b int [0, 255]
    '''
    x, y, z = rgb2xyz(r, g, b)
    L, a, b = xyz2lab(x, y, z)

    return L, a, b