import torch
import torch.nn as nn
import torch.nn.functional as F
from models.unet import UNet
import utils


class GaussianDiffusion(utils.Module):
    
    def __init__(
        self,
        args,
    ):
        super().__init__()
        
        # member variables
        unet_args = args['unet']
        self.denoise_fn = UNet(**unet_args)
        self.time_steps = args['betas']['time_step']

        # parameters 
        scale = 1000 / self.time_steps
        betas = torch.linspace(scale * args['betas']['linear_start'], scale * args['betas']['linear_end'], self.time_steps, dtype=torch.float32)
        alphas = 1. - betas
        gammas = torch.cumprod(alphas, axis=0)
        gammas_prev = F.pad(gammas[:-1], (1, 0), value=1.)

        # diffusion q(x_t | x_{t-1}) 
        self.register_buffer('gammas', gammas)
        self.register_buffer('sqrt_reciprocal_gammas', torch.sqrt(1. / gammas))
        self.register_buffer('sqrt_reciprocal_gammas_m1', torch.sqrt(1. / gammas - 1))

        # posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - gammas_prev) / (1. - gammas)
        self.register_buffer('posterior_log_variance', torch.log(torch.clamp(posterior_variance, 1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(gammas_prev) / (1. - gammas))
        self.register_buffer('posterior_mean_coef2', (1. - gammas_prev) * torch.sqrt(alphas) / (1. - gammas))

        # loss function
        self.loss_fn = F.mse_loss


    def forward(self, x, t):
        """
        :param[in]  x   torch.Tensor    [batch_size x channel x height x weight]
        :param[in]  t   torch.Tensor    [batch_size]
        """
        assert x.shape[0] == t.shape[0]
        
        # noise
        noise = torch.randn_like(x)

        # q sampling
        gammas_t = utils.extract(self.gammas, t, x_shape=x.shape)
        x_noisy = torch.sqrt(gammas_t) * x + torch.sqrt(1 - gammas_t) * noise


        return 0