import torch.utils.data
import torch.utils.data.distributed
import utils.pythonic
import data


class DataHandler(object):
    def __init__(
        self,
        name,
        global_rank=1,
        **kwargs
    ):
        # Dataset
        dataset_class = getattr(data, name)
        dataset_option = utils.pythonic.argument_autofilling(dataset_class.__init__, kwargs=kwargs)
        self.dataset = dataset_class(**dataset_option)
        # Distribution
        self.sampler = None
        if global_rank > 1:
            self.sampler = torch.utils.data.distributed.DistributedSampler(self.dataset)
            kwargs['shuffle'] = False
        # Data loader
        dataloader_option = dict()
        for arg_name in torch.utils.data.DataLoader.__init__.__code__.co_varnames:
            if arg_name in kwargs:
                dataloader_option[arg_name] = kwargs[arg_name]
        if global_rank > 1:
            dataloader_option['batch_size'] = int(dataloader_option.get('batch_size', 1) / global_rank)
            dataloader_option['num_workers'] = int(dataloader_option.get('num_workers', 4) / global_rank)
        self.data_loader = torch.utils.data.DataLoader(dataset=self.dataset, sampler=self.sampler, **dataloader_option)
