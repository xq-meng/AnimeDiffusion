import os
import base
import torch
import utils.image
import utils.path


class TestAnimeDiffusion(base.Tester):
    def preprocessing(self):
        super().preprocessing()
        self.output_dir = self._kwargs['output_dir']
        utils.path.mkdir(self.output_dir)

    def step(self, data):
        x_ref = data['reference'].to(self.device)
        x_con = data['condition'].to(self.device)
        x_dis = data['distorted'].to(self.device)

        noise = torch.randn_like(x_ref).to(self.device)
        with torch.no_grad():
            rets = self.model_handler.model.inference_ddim(x_t=noise, x_cond=torch.cat([x_con, x_dis], dim=1))[-1]
        images = utils.image.tensor2PIL(rets)
        for i, filename in enumerate(data['name']):
            output_path = os.path.join(self.output_dir, 'ret_' + filename)
            images[i].save(output_path)
            self.logger.info(f'Test output saved as {output_path}')