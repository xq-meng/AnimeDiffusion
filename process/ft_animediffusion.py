import base
import torch
import torch.nn.functional as F


class FineTuneAnimeDiffusion(base.Trainer):
    def preprocessing(self):
        super().preprocessing()
        self.time_steps = self.model_handler.model.time_steps

    def step(self, data):
        # data
        x_ref = data['reference'].to(self.device)
        x_con = data['condition'].to(self.device)
        x_dis = data['distorted'].to(self.device)

        with torch.no_grad():
            x_T = self.model_handler.model.fix_forward(x_ref, x_cond=torch.cat([x_con, x_dis], dim=1))

        x_til = self.model_handler.model.inference_ddim(x_t=x_T, x_cond=torch.cat([x_con, x_dis], dim=1))[-1]

        return F.mse_loss(x_til, x_ref)