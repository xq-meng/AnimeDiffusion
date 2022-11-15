import argparse
import json
import os
from torch.utils.data import DataLoader
import utils
from models.palette import Palette
import datasets


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as fr:
            config = json.load(fr)
    except FileNotFoundError:
        print("No such file: {}".format(args.config))
        exit()
    except PermissionError:
        print("Unable to access: {}".format(args.config))
        exit()

    # logger initialize
    logger = utils.Logger(name='base_logger', **config['logging'])

    # dataset
    train_dataset = datasets.ColorizationDataset(**config['dataset']['train']['path'])
    train_data_loader = DataLoader(dataset=train_dataset ,**config['dataset']['train']['dataloader'])
    test_dataset = datasets.ColorizationDataset(**config['dataset']['test']['path'])
    test_data_loader = DataLoader(dataset=test_dataset, **config['dataset']['test']['dataloader'])

    # palette
    model = Palette(config['model'], logger=logger)
    model.train(train_epochs=config['train']['epochs'], data_loader=train_data_loader)

    # inference
    utils.mkdir(config['test']['output_dir'])
    model.inference(data_loader=test_dataset, output_dir=config['test']['output_dir'])