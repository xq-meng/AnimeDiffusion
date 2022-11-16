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
        self.noise_channel = args['diffusion']['unet']['channel_out']
        self.diffusion_model = GaussianDiffusion(args=args['diffusion'])
        self.diffusion_model.to(device=self.device)
        # optimizer
        opt_learning_rate = args['optimizer']['learning_rate'] if 'learning_rate' in args['optimizer'] else 5e-4
        opt_beta1 = args['optimizer']['beta1'] if 'beta1' in args['optimizer'] else 0.9
        opt_beta2 = args['optimizer']['beta2'] if 'beta2' in args['optimizer'] else 0.999

        if args['optimizer']['name'] == 'SGD':
            self.optimizer = torch.optim.SGD(self.diffusion_model.parameters(), lr=opt_learning_rate)
        else:
            self.optimizer = torch.optim.Adam(self.diffusion_model.parameters(), lr=opt_learning_rate, betas=(opt_beta1, opt_beta2))
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

    def train(self, train_epochs, data_loader, validation=None):
        while self.epoch < train_epochs:
            # train step
            for step, images in enumerate(data_loader):
                self.optimizer.zero_grad()
                x_ref = images['reference']
                x_con = images['condition']
                x_ref = x_ref.to(self.device)
                x_con = x_con.to(self.device)
                batch_size = x_ref.shape[0]
                t = torch.randint(0, self.diffusion_model.time_steps, (batch_size, ), device=self.device).long()
                self.loss = self.diffusion_model.train(x=x_ref, t=t, x_cond=x_con)
                if step % 200 == 0:
                    self._logger.info("Epoch = %d, Loss = %f", self.epoch, self.loss)
                self.loss.backward()
                self.optimizer.step()
                self.ema.update()
            # save status
            if self.status_save_dir is not None and self.epoch % self.status_save_epochs == 0:
                self.save_status(os.path.join(self.status_save_dir, 'epoch_' + str(self.epoch).zfill(5) + '.pkl'))
            # mid validation
            if validation is not None:
                v_con = validation['condition']
                v_con = v_con.to(self.device)
                v_output = os.path.join(validation['output_dir'], 'valid_epoch_' + str(self.epoch).zfill(5) + validation['postfix'])
                self.inference(x_con=v_con, output_path=v_output)
            # update epoch
            self.epoch += 1
        # save final status
        if self.status_save_dir is not None:
            self.save_status(os.path.join(self.status_save_dir, 'trained.pkl'))

    def inference(self, data_loader, output_dir=None):
        self.ema.apply_shadow()
        rets = []
        for step, images in enumerate(data_loader):
            x_con = images['condition']
            output_path = None if output_dir is None else os.path.join(output_dir, 'ret_' + images['name'])
            x_ret = self.inference(x_con=x_con, output_path=output_path)
            rets.append(x_ret)
        self.ema.restore()
        return rets

    def inference(self, x_con: torch.Tensor, output_path=None):
        batch_size, _, h, w = x_con.shape
        noise = torch.randn((batch_size, self.noise_channel, h, w))
        noise = noise.to(self.device)
        x_ret = self.diffusion_model.inference(noise, x_cond=x_con)
        if output_path is not None:
            x_pils = utils.tensor2PIL(x_ret[-1])
            for x_pil in x_pils:
                x_pil.save(output_path)
        return x_ret
