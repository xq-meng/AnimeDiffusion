import os
import logging
import torch
from torch.utils.data import DataLoader
from models.diffusion import GaussianDiffusion


class Palette:

    def __init__(self, args):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # training
        self.epoch = 0
        self.loss = 1e+5
        self.train_epochs = args['train']['epochs']
        self.batch_size = args['train']['batch_size']
        self.ema_decay = args['train']['ema_decay']
        learning_rate = args['train']['learning_rate'] if 'learning_rate' in args['train'] else 5e-4

        # gaussian diffusion
        self.diffusion_model = GaussianDiffusion(args=args['diffusion'])
        self.diffusion_model.to(device=self.device)
        self.optimizer = torch.optim.Adam(self.diffusion_model.parameters(), lr=learning_rate)

        # logging
        logging.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.DEBUG)

        # file path
        self.status_save_dir = args['status_save_dir']

    def save_status(self, path_to_status):
        torch.save({
            'epoch': self.epoch,
            'diffusion_state_dict': self.diffusion_model.state_dict(),
            'loss': self.loss},
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

    def train(self, dataset):
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        while self.epoch < self.train_epochs:
            self.epoch += 1
            # save status
            if self.epoch > 0 and self.epoch % 20 == 0:
                self.save_status(os.path.join(self.status_save_dir, 'epoch_' + str(self.epoch).zfill(5) + '.pkl'))
            # train step
            for step, (images, _) in enumerate(data_loader):
                self.optimizer.zero_grad()
                batch_size = images.shape[0]
                x = images[:, :1, :, :]
                x_cond = images[:, 1:, :, :]
                x = x.to(self.device)
                x_cond = x_cond.to(self.device)
                t = torch.randint(0, self.diffusion_model.time_steps, (batch_size, ), device=self.device).long()
                self.loss = self.diffusion_model.train(x=x, t=t, x_cond=x_cond)
                if step % 200 == 0:
                    logging.info("Epoch = %d, Loss = %f", self.epoch, self.loss)
                self.loss.backward()
                self.optimizer.step()

    def inference(self, dataset):
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        rets = []
        for step, (images, _) in enumerate(data_loader):
            batch_size, _, h, w = images.shape
            x_cond = images[:, 1:, :, :]
            noise = torch.randn((batch_size, 1, h, w))
            rets.append(self.diffusion_model.inference(noise, x_cond=x_cond))

        return rets