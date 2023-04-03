import torch
from abc import abstractclassmethod
from base.model_handler import ModelHandler
from base.data_handler import DataHandler
import utils.logger


class Tester(object):
    def __init__(
        self,
        datas_option,
        model_option=None,
        local_rank=0,
        global_rank=1,
        logger_option=None,
        **kwargs
    ):
        # distribution
        self.local_rank = local_rank if local_rank >= 0 and torch.cuda.is_available() else -1
        self.global_rank = global_rank if self.local_rank >= 0 else 0
        self.device = f'cuda:{self.local_rank}' if self.local_rank >= 0 else 'cpu'
        # model
        self.model_handler = ModelHandler(local_rank=local_rank, global_rank=global_rank, **model_option) if model_option is not None else None
        # data
        self.data_handler = DataHandler(global_rank=global_rank, **datas_option)
        # logger
        logger_name = 'rank' + str(self.local_rank)
        self.logger = utils.logger.Logger(name=logger_name, **logger_option) if logger_option is not None else utils.logger.Logger(name=logger_name)
        # kwargs
        self._kwargs = kwargs

    def run(self):
        self.preprocessing()
        for step, data in enumerate(self.data_handler.data_loader):
            self.step(data)

    def preprocessing(self):
        if self.model_handler:
            self.model_handler.train(mode=False)
        return

    @abstractclassmethod
    def step(self, data):
        raise NotImplemented