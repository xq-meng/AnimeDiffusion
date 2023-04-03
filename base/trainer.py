import os
from abc import abstractclassmethod
import torch
from base.data_handler import DataHandler
from base.model_handler import ModelHandler
from base.optimizer_handler import OptimizerHandler
import utils.path
import utils.logger


class Trainer(object):
    def __init__(
        self,
        model_option,
        optim_option,
        datas_option,
        local_rank=0,
        global_rank=1,
        start_epoch=0,
        target_epoch=-1,
        accumulate_step=1,
        checkpoint_epoch=20,
        checkpoint_dir=None,
        logger_option=None,
        logging_step=200,
        **kwargs
    ) -> None:
        # training config
        self.target_epoch = target_epoch
        self.accumulate_step = accumulate_step
        self.epoch = start_epoch
        # distribution
        self.local_rank = local_rank if local_rank >= 0 and torch.cuda.is_available() else -1
        self.global_rank = global_rank if self.local_rank >= 0 else 0
        self.device = f'cuda:{self.local_rank}' if self.local_rank >= 0 else 'cpu'
        # model
        self.model_handler = ModelHandler(local_rank=local_rank, global_rank=global_rank, **model_option)
        # data
        self.data_handler = DataHandler(global_rank=global_rank, **datas_option)
        # optimizer
        self.optim_handler = OptimizerHandler(parameters=self.model_handler.get('parameters')(), **optim_option)
        # checkpoints
        self.checkpoint_epoch = checkpoint_epoch if checkpoint_dir is not None else 10000
        self.checkpoint_dir = checkpoint_dir
        if self.local_rank <= 0:
            utils.path.mkdir(self.checkpoint_dir)
        # logger
        logger_name = 'rank' + str(self.local_rank)
        self.logger = utils.logger.Logger(name=logger_name, **logger_option) if logger_option is not None else utils.logger.Logger(name=logger_name)
        self.logging_step = logging_step
        # kwargs
        self._kwargs = kwargs

    def run(self):
        self.preprocessing()
        while self.epoch < self.target_epoch:
            if self.global_rank > 1:
                self.data_handler.sampler.set_epoch(self.epoch)
            avg_loss = 0.
            optim_step_flag = True
            for step, data in enumerate(self.data_handler.data_loader):
                # update optimizer
                self.optim_handler.update_lr(epoch=(step / len(self.data_handler.data_loader) + self.epoch), target_epoch=self.target_epoch)
                # model training step
                loss = self.step(data)
                if torch.isnan(loss).int().sum() > 0:
                    self.logger.warning(f'Rank = {self.local_rank}, Epoch = {self.epoch}, Step = {step}. Loss is nan! ignored!')
                    optim_step_flag = False
                avg_loss += loss
                # loss regularization
                loss = loss / self.accumulate_step
                loss.backward()
                # optimizer
                if step % self.accumulate_step == 0:
                    if optim_step_flag:
                        self.optim_handler.optimizer.step()
                    self.optim_handler.optimizer.zero_grad()
                    optim_step_flag = True
                if step % self.logging_step == 0:
                    avg_loss = avg_loss / self.logging_step if step > 0 else avg_loss
                    self.logger.info(f'Rank = {self.local_rank}, Epoch = {self.epoch}, LR = {self.optim_handler.learning_rate():.4e}, Loss = {avg_loss:.4e}')
                    avg_loss = 0.
            if (self.local_rank <= 0) and (self.checkpoint_dir is not None) and ((self.epoch + 1) % self.checkpoint_epoch == 0):
                model_checkpoint_filename = os.path.join(self.checkpoint_dir, f'ep_{self.epoch}.model.pth')
                optim_checkpoint_filename = os.path.join(self.checkpoint_dir, f'ep_{self.epoch}.optim.pth')
                self.model_handler.save_checkpoint(model_checkpoint_filename)
                self.optim_handler.save_checkpoint(optim_checkpoint_filename)
            # update epoch
            self.epoch += 1
        if (self.local_rank <= 0) and (self.checkpoint_dir is not None):
            model_checkpoint_filename = os.path.join(self.checkpoint_dir, 'trained.model.pth')
            optim_checkpoint_filename = os.path.join(self.checkpoint_dir, 'trained.optim.pth')
            self.model_handler.save_checkpoint(model_checkpoint_filename)
            self.optim_handler.save_checkpoint(optim_checkpoint_filename)

    def preprocessing(self):
        self.model_handler.train()
        return

    @abstractclassmethod
    def step(self, data):
        raise NotImplemented
