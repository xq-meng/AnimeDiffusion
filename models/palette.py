import os
import csv
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
        self.lr_deacy = args['optimizer']['decay'] if 'decay' in args['optimizer'] else -1.0
        if args['optimizer']['name'] == 'SGD':
            self.optimizer = torch.optim.SGD(self.diffusion_model.parameters(), lr=opt_learning_rate)
        else:
            self.optimizer = torch.optim.Adam(self.diffusion_model.parameters(), lr=opt_learning_rate, betas=(opt_beta1, opt_beta2))
        if self.lr_deacy > 0:
            self.opt_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=self.lr_deacy)
        # ema
        self.ema_decay = min(max(args['ema']['decay'], 0), 1)
        self.ema = utils.ema(self.diffusion_model.denoise_fn, self.ema_decay)
        self.ema.register()
        # logger
        self._logger = logger if logger is not None else utils.logger()
        # status save directory
        self.status_save_epochs = args['status']['save_epochs']
        self.status_save_dir = args['status']['save_dir']
        # load status from file
        if 'load_path' in args['status'] and os.access(args['status']['load_path'], os.R_OK):
            self.load_status(args['status']['load_path'])

    def save_status(self, path_to_status):
        torch.save({
            'epoch': self.epoch,
            'diffusion_state_dict': self.diffusion_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
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
        self.optimizer.load_state_dict(a_status['optimizer_state_dict'])
        self.ema.shadow = a_status['ema_shadow']
        self.ema.backup = a_status['ema_backup']
        self._logger.info('Load status from %s', path_to_status)
        self._logger.info('Current epoch : %d', self.epoch)
        self._logger.info('Current loss : %f', self.loss)

    def train(self, epochs, data_loader, validations=[], **kwargs):
        utils.mkdir(self.status_save_dir)
        while self.epoch < epochs:
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
            for vi, validation in enumerate(validations):
                v_con = validation['condition']
                v_con = v_con.to(self.device)
                v_output = os.path.join(validation['output_dir'], 'valid_epoch_' + str(self.epoch).zfill(5) + '_' + str(vi) + '_' + validation['filename'])
                with torch.no_grad():
                    v_ret = self.inference(v_con)[-1]
                v_pil = utils.tensor2PIL(v_ret)[0]
                v_pil.save(v_output)
            # update epoch
            self.epoch += 1
            # update learning rate
            if self.lr_deacy > 0:
                self.opt_lr_scheduler.step()
                self._logger.info("Epoch = %d, Learning Rate = %f", self.epoch, self.optimizer.state_dict()['param_groups'][0]['lr'])
        # save final status
        if self.status_save_dir is not None:
            self.save_status(os.path.join(self.status_save_dir, 'trained.pkl'))

    def inference(self, x_con: torch.Tensor, eta=1, noise=None, use_ddim=False, sample_steps=10):
        batch_size, _, h, w = x_con.shape
        if noise is None:
            noise = torch.randn((batch_size, self.noise_channel, h, w))
        noise = noise.to(self.device)
        if use_ddim:
            with torch.no_grad():
                ret = self.diffusion_model.inference_ddim(noise, time_steps=sample_steps, x_cond=x_con, eta=eta)
        else:
            with torch.no_grad():
                ret = self.diffusion_model.inference(noise, x_cond=x_con, eta=eta)
        return ret

    def test(self, data_loader, output_dir, use_ddim=False, sample_steps=10, noise_init=True, **kwargs):
        for step, images in enumerate(data_loader):
            x_cons = images['condition']
            x_cons = x_cons.to(self.device)
            if noise_init:
                with torch.no_grad():
                    x_noise = self.diffusion_model.unseen_transform(x_cons[:, 1:, :, :].to(self.device))
            else:
                x_noise = None
            x_rets = self.inference(x_con=x_cons, noise=x_noise, eta=1, use_ddim=use_ddim, sample_steps=sample_steps)[-1]
            x_pils = utils.tensor2PIL(x_rets)
            for i, filename in enumerate(images['name']):
                output_path = os.path.join(output_dir, 'ret_' + filename)
                x_pils[i].save(output_path)
                self._logger.info("Test output saved as {0}".format(output_path))

    def fine_tune(self, epochs, data_loader, validations=[], eta=1, **kwargs):
        utils.mkdir(self.status_save_dir)
        ep = 0
        while ep < epochs:
            for step, images in enumerate(data_loader):
                self.optimizer.zero_grad()
                x_ref = images['reference']
                x_con = images['condition']
                x_ref = x_ref.to(self.device)
                x_con = x_con.to(self.device)
                with torch.no_grad():
                    noise = self.diffusion_model.unseen_transform(x_con[:, 1:, :, :]).to(self.device)
                loss = self.diffusion_model.fine_tune(x=x_ref, x_t=noise, x_cond=x_con, eta=eta)
                if step % 200 == 0:
                    self._logger.info("Fine tune epoch = %d, Loss = %f", ep, loss)
                loss.backward()
                self.optimizer.step()
            if self.status_save_dir is not None and ep % self.status_save_epochs == 0:
                self.save_status(os.path.join(self.status_save_dir, 'ep_' + str(self.epoch).zfill(4) + '_ft_' + str(ep).zfill(4) + '.pkl'))
            # mid validation
            for vi, validation in enumerate(validations):
                v_con = validation['condition']
                v_con = v_con.to(self.device)
                v_output = os.path.join(validation['output_dir'], 'valid_ep_' + str(self.epoch).zfill(4) + '_ft_' +str(ep).zfill(4) + '_' + str(vi) + '_' + validation['filename'])
                with torch.no_grad():
                    v_ret = self.inference(v_con)[-1]
                v_pil = utils.tensor2PIL(v_ret)[0]
                v_pil.save(v_output)
            ep += 1
        # save final status
        if self.status_save_dir is not None:
            self.save_status(os.path.join(self.status_save_dir, 'ep_' + str(self.epoch).zfill(4) + '_ft_' + str(ep).zfill(4) + '.pkl'))

    def find_lr(self, data_loader, lr_start, lr_end, beta=0.98, lr_path=None, **kwargs):
        def adopt_lr(optimizer, lr):
            for parameter_group in self.optimizer.param_groups:
                parameter_group['lr'] = lr

        rets = []
        lr = lr_start
        n = len(data_loader.dataset) // data_loader.batch_size
        q = pow(lr_end / lr_start, 1.0 / n)
        avg_loss = 0
        for step, images in enumerate(data_loader):
            self.optimizer.zero_grad()
            adopt_lr(self.optimizer, lr)
            x_ref = images['reference']
            x_con = images['condition']
            x_ref = x_ref.to(self.device)
            x_con = x_con.to(self.device)
            batch_size = x_ref.shape[0]
            t = torch.randint(0, self.diffusion_model.time_steps, (batch_size, ), device=self.device).long()
            loss = self.diffusion_model.train(x=x_ref, t=t, x_cond=x_con)
            avg_loss = beta * avg_loss + (1 - beta) * loss
            smoothed_loss = avg_loss / (1 - beta ** (step + 1))
            self._logger.info("Step = %d, Loss = %f, Smoothed Loss: %f", step, loss, smoothed_loss)
            rets.append([lr, loss, smoothed_loss])
            loss.backward()
            self.optimizer.step()
            lr = lr * q

        if lr_path is not None:
            with open(lr_path, 'wb') as f:
                csv_write = csv.writer(f)
                csv_head = ["step", "loss", "smoothed_loss"]
                csv_write.writerow(csv_head)
                csv_write.writerows(rets)
                f.close()