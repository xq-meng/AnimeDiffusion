import os
import torch
import utils
from models.diffusion import GaussianDiffusion


class Palette:

    def __init__(self, args, logger=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.epoch = 0
        self.loss = 1e+5
        # gaussian diffusion
        self.diffusion_model = GaussianDiffusion(args=args['diffusion'])
        self.diffusion_model.to(device=self.device)
        # optimizer
        learning_rate = args['optimizer']['learning_rate'] if 'learning_rate' in args['optimizer'] else 5e-4
        if args['optimizer']['name'] == 'SGD':
            self.optimizer = torch.optim.SGD(self.diffusion_model.parameters(), lr=learning_rate)
        else:
            self.optimizer = torch.optim.Adam(self.diffusion_model.parameters(), lr=learning_rate)
        # ema
        self.ema_decay = min(max(args['ema']['decay'], 0), 1)
        self.ema = utils.ema(self.diffusion_model.denoise_fn, self.ema_decay)
        self.ema.register()
        # logger
        self._logger = logger if logger is not None else utils.logger()
        # status save directory
        self.status_save_epochs = args['status']['save_epochs']
        self.status_save_dir = args['status']['save_dir']
        utils.mkdir(self.status_save_dir)
        # load status from file
        if 'load_path' in args['status'] and os.access(args['status']['load_path'], os.R_OK):
            self.load_status(args['status']['load_path'])

    def save_status(self, path_to_status):
        torch.save({
            'epoch': self.epoch,
            'diffusion_state_dict': self.diffusion_model.state_dict(),
            'loss': self.loss,
            'ema_shadow': self.ema.shadow,
            'ema_backup': self.ema.backup},
            path_to_status
        )
        self._logger.info('Save status as %s', path_to_status)

    def load_status(self, path_to_status):
        a_status = torch.load(path_to_status)
        self.epoch = a_status['epoch']
        self.loss = a_status['loss']
        self.diffusion_model.load_state_dict(a_status['diffusion_state_dict'])
        self._logger.info('Load status from %s', path_to_status)
        self._logger.info('Current epoch : %d', self.epoch)
        self._logger.info('Current loss : %f', self.loss)

    def train(self, train_epochs, data_loader):
        while self.epoch < train_epochs:
            # save status
            if self.status_save_dir is not None and self.epoch % self.status_save_epochs == 0:
                self.save_status(os.path.join(self.status_save_dir, 'epoch_' + str(self.epoch).zfill(5) + '.pkl'))
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
                    self._logger.info("Epoch = %d, Loss = %f", self.epoch, self.loss)
                self.loss.backward()
                self.optimizer.step()
                self.ema.update()
            # update epoch
            self.epoch += 1
        # save final status
        if self.status_save_dir is not None:
            self.save_status(os.path.join(self.status_save_dir, 'trained.pkl'))

    def inference(self, data_loader, output_dir=None):
        self.ema.apply_shadow()
        rets = []
        for step, (images, _) in enumerate(data_loader):
            batch_size, _, h, w = images.shape
            x_cond = images[:, :1, :, :]
            noise = torch.randn((batch_size, 2, h, w))
            image_ab = self.diffusion_model.inference(noise, x_cond=x_cond)
            image_lab = torch.cat([x_cond, image_ab], dim=1)
            image_rgb = utils.Postprocess()(image_lab)
            if output_dir is not None:
                image_rgb.save(os.path.join(output_dir, str(step).zfill(5) + '.jpg'))
            else:
                rets.append(image_rgb)
        self.ema.restore()
        return rets