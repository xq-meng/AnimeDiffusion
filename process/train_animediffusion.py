import base
import torch
import torch.nn.functional as F


class TrainAnimeDiffusion(base.Trainer):
    def preprocessing(self):
        super().preprocessing()
        self.time_steps = self.model_handler.model.time_steps

    def step(self, data):
        # data
        batch_size = len(data['name'])
        x_ref = data['reference'].to(self.device)
        x_con = data['condition'].to(self.device)
        x_dis = data['distorted'].to(self.device)

        # forward process
        t = torch.randint(0, self.time_steps, (batch_size, ), device=self.device).long()
        return self.model_handler(x=x_ref, t=t, x_cond=torch.cat([x_con, x_dis], dim=1))