import os
import logging
import torch
from torch.utils.data import DataLoader
import utils
from models.diffusion import GaussianDiffusion


class Palette:

    def __init__(self, args):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # training
        self.epoch = 0
        self.loss = 1e+5
        self.train_epochs = args['train']['epochs']
        self.batch_size = args['train']['batch_size']
        self.ema_decay = min(max(args['train']['ema_decay'], 0), 1)
        self.save_epochs = args['train']['save_epochs']
        learning_rate = args['train']['learning_rate'] if 'learning_rate' in args['train'] else 5e-4

        # gaussian diffusion
        self.diffusion_model = GaussianDiffusion(args=args['diffusion'])
        self.diffusion_model.to(device=self.device)
        self.optimizer = torch.optim.Adam(self.diffusion_model.parameters(), lr=learning_rate)

        # ema
        self.ema = utils.ema(self.diffusion_model.denoise_fn, self.ema_decay)
        self.ema.register()

    def save_status(self, path_to_status):
        torch.save({
            'epoch': self.epoch,
            'diffusion_state_dict': self.diffusion_model.state_dict(),
            'loss': self.loss,
            'ema_shadow': self.ema.shadow,
            'ema_backup': self.ema.backup},
            path_to_status
        )
        logging.info('Save status as %s', path_to_status)

    def load_status(self, path_to_status):
        a_status = torch.load(path_to_status)
        self.epoch = a_status['epoch']
        self.loss = a_status['loss']
        self.diffusion_model.load_state_dict(a_status['diffusion_state_dict'])
        logging.info('Load status from %s', path_to_status)
        logging.info('Current epoch : %d', self.epoch)
        logging.info('Current loss : %f', self.loss)

    def train(self, dataset, status_save_dir=None):
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        while self.epoch < self.train_epochs:
            # save status
            if status_save_dir is not None and self.epoch % self.save_epochs == 0:
                self.save_status(os.path.join(status_save_dir, 'epoch_' + str(self.epoch).zfill(5) + '.pkl'))
            # train step
            for step, (images, _) in enumerate(data_loader):
                self.optimizer.zero_grad()
                batch_size = images.shape[0]
                # prediction: 2nd, 3rd channel
                x = images[:, 1:, :, :]
                # condition: 1st channel
                x_cond = images[:, :1, :, :]
                x = x.to(self.device)
                x_cond = x_cond.to(self.device)
                t = torch.randint(0, self.diffusion_model.time_steps, (batch_size, ), device=self.device).long()
                self.loss = self.diffusion_model.train(x=x, t=t, x_cond=x_cond)
                if step % 200 == 0:
                    logging.info("Epoch = %d, Loss = %f", self.epoch, self.loss)
                self.loss.backward()
                self.optimizer.step()
                self.ema.update()
            # update epoch
            self.epoch += 1
        # save final status
        self.save_status(os.path.join(self.status_save_dir, 'trained.pkl'))

    def inference(self, dataset, output_dir=None):
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self.ema.apply_shadow()
        rets = []
        for step, (images, _) in enumerate(data_loader):
            batch_size, _, h, w = images.shape
            x_cond = images[:, :1, :, :]
            noise = torch.randn((batch_size, 2, h, w))
            image_lab = self.diffusion_model.inference(noise, x_cond=x_cond)
            image_rgb = utils.Postprocess()(image_lab)
            if output_dir is not None:
                image_rgb.save(os.path.join(output_dir, str(step).zfill(5) + '.jpg'))
            else:
                rets.append(image_rgb)
        self.ema.restore()
        return rets