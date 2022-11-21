import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.unet import UNet
import utils


class GaussianDiffusion(nn.Module):

    def __init__(
        self,
        args,
    ):
        super().__init__()

        # member variables
        self.denoise_fn = UNet(**args['unet'])
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
        gammas_t = utils.extract(self.gammas, t, x_shape=x_0.shape).to(x_0.device)
        return torch.sqrt(gammas_t) * x_0 + torch.sqrt(1 - gammas_t) * noise

    @torch.no_grad()
    def p_sample(self, x_t, t, x_cond=None, eta=1):
        predicted_noise = self.denoise_fn(x_t, t) if x_cond is None else self.denoise_fn(torch.cat([x_t, x_cond], dim=1), t)
        predicted_x_0 = utils.extract(self.sqrt_reciprocal_gammas, t, x_t.shape) * x_t - utils.extract(self.sqrt_reciprocal_gammas_m1, t, x_t.shape) * predicted_noise
        predicted_x_0 = torch.clamp(predicted_x_0, min=-1., max=1.)
        posterior_mean = utils.extract(self.posterior_mean_coef1, t, x_t.shape) * predicted_x_0 + utils.extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        posterior_log_variance = utils.extract(self.posterior_log_variance, t, x_t.shape)
        noise = torch.randn_like(x_t)
        nonzero_mask = eta * ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        pred_x = posterior_mean + nonzero_mask * (0.5 * posterior_log_variance).exp() * noise
        return pred_x

    @torch.no_grad()
    def inference(self, x_t, x_cond=None, eta=1):
        batch_size = x_t.shape[0]
        device = next(self.parameters()).device
        ret = []
        for i in reversed(range(0, self.time_steps)):
            x_t = self.p_sample(x_t=x_t, t=torch.full((batch_size, ), i, device=device, dtype=torch.long), x_cond=x_cond, eta=eta)
            ret.append(x_t.cpu())
        return ret

    def inference_ddim(self, x_t, time_steps=100, x_cond=None, eta=1):
        step_sequence = np.asarray(list(range(0, self.time_steps, self.time_steps // time_steps)))
        step_sequence = step_sequence + 1
        step_sequence_prev = np.append(np.array([0]), step_sequence[:-1])
        batch_size = x_t.shape[0]
        device = next(self.parameters()).device
        ret = []
        for i in reversed(range(0, time_steps)):
            t = torch.full((batch_size, ), step_sequence[i], device=device, dtype=torch.long)
            prev_t = torch.full((batch_size, ), step_sequence_prev[i], device=device, dtype=torch.long)
            gammas_t = utils.extract(self.gammas, t, x_shape=x_t.shape)
            gammas_prev_t = utils.extract(self.gammas, prev_t, x_shape=x_t.shape)
            predicted_noise = self.denoise_fn(x_t, t) if x_cond is None else self.denoise_fn(torch.cat([x_t, x_cond], dim=1), t)
            predicted_x_0 = (x_t - torch.sqrt(1. - gammas_t) * predicted_noise) / torch.sqrt(gammas_t)
            predicted_x_0 = torch.clamp(predicted_x_0, min=-1., max=1.)
            posterier_varience = (1. - gammas_prev_t) / (1. - gammas_t) * (1. - gammas_t / gammas_prev_t)
            sigma_t = eta * torch.sqrt(posterier_varience)
            predicted_x_t_direction = torch.sqrt(1. - gammas_prev_t - sigma_t**2) * predicted_noise
            noise = torch.randn_like(x_t)
            x_t = torch.sqrt(gammas_prev_t) * predicted_x_0 + predicted_x_t_direction + sigma_t * noise
            ret.append(x_t)
        return ret

    @torch.no_grad()
    def unseen_transform(self, x_0):
        batch_size = x_0.shape[0]
        t = (self.time_steps - 1) * torch.ones(batch_size).long()
        gammas_t = utils.extract(self.gammas, t, x_shape=x_0.shape).to(x_0.device)
        noise = torch.randn_like(x_0)
        x_noise = torch.sqrt(gammas_t) * x_0 + torch.sqrt(1 - gammas_t) * noise
        return torch.sqrt(gammas_t) * x_noise + torch.sqrt(1 - gammas_t) * noise

    def train(self, x, t, x_cond=None):
        """
        :param[in]  x       torch.Tensor    [batch_size x channel x height x weight]
        :param[in]  t       torch.Tensor    [batch_size]
        :param[in]  x_cond  torch.Tensor    [batch_size x _ x height x weight]
        """
        assert x.shape[0] == t.shape[0]
        # noise
        noise = torch.randn_like(x)
        # q sampling
        x_noisy = self.q_sample(x, t, noise=noise)
        # add denoise condition
        if x_cond is not None:
            assert x.shape[0] == x_cond.shape[0] and x.shape[2] == x_cond.shape[2] and x.shape[3] == x_cond.shape[3]
            x_noisy = torch.cat([x_noisy, x_cond], dim=1)
        # noise prediction
        noise_tilde = self.denoise_fn(x_noisy, t)
        return self.loss_fn(noise, noise_tilde)

    def fine_tune(self, x, x_t=None, x_cond=None):
        if x_t is None:
            x_t = torch.randn_like(x)
        assert x.shape == x_t.shape
        # ddim inference
        x_tilde = self.inference_ddim(x_t, self.time_steps // 50, x_cond=x_cond)
        return self.loss_fn(x, x_tilde)