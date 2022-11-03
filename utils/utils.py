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