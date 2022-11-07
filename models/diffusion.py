import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from models.unet import UNet
import utils


class GaussianDiffusion(nn.Module):
    
    def __init__(
        self,
        args,
    ):
        super().__init__()
        
        # member variables
        unet_args = args['unet']
        self.denoise_fn = UNet(**unet_args)
        self.time_steps = args['time_step']

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
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance', torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(gammas_prev) / (1. - gammas))
        self.register_buffer('posterior_mean_coef2', (1. - gammas_prev) * torch.sqrt(alphas) / (1. - gammas))

        # loss function
        self.loss_fn = F.mse_loss

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        gammas_t = utils.extract(self.gammas, t, x_shape=x_0.shape)
        return torch.sqrt(gammas_t) * x_0 + torch.sqrt(1 - gammas_t) * noise

    def p_sample(self, x_t, t):
        predicted_noise = self.denoise_fn(x_t, t)
        predicted_x_0 = utils.extract(self.sqrt_reciprocal_gammas, t, x_t.shape) * x_t - utils.extract(self.sqrt_reciprocal_gammas_m1, t, x_t.shape) * predicted_noise
        predicted_x_0 = torch.clamp(predicted_x_0, min=-1., max=1.)
        posterior_mean = utils.extract(self.posterior_mean_coef1, t, x_t.shape) * predicted_x_0 + utils.extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        posterior_log_variance = utils.extract(self.posterior_log_variance, t, x_t.shape)
        noise = torch.randn_like(x_t)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        pred_x = posterior_mean + nonzero_mask * (0.5 * posterior_log_variance).exp() * noise
        return pred_x
    
    def inference(self, x_t):
        batch_size = x_t.shape[0]
        device = next(self.parameters()).device
        ret = []
        for i in tqdm(reversed(range(0, self.time_steps)), desc='Sampling time step', total=self.time_steps):
            x_t = self.p_sample(x_t=x_t, t=torch.full((batch_size, ), i, device=device, dtype=torch.long))
            ret.append(x_t.cpu().numpy())
        return ret

    def train(self, x, t):
        """
        :param[in]  x   torch.Tensor    [batch_size x channel x height x weight]
        :param[in]  t   torch.Tensor    [batch_size]
        """
        assert x.shape[0] == t.shape[0]
        # noise
        noise = torch.randn_like(x)
        # q sampling
        x_noisy = self.q_sample(x, t, noise=noise)
        # noise prediction
        noise_tilde = self.denoise_fn(x_noisy, t)
        return self.loss_fn(noise, noise_tilde)