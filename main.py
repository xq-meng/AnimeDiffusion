import os
import argparse
import json
from torch.utils.data import DataLoader
from PIL import Image
import utils
from models.palette import Palette
import datasets


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-x', '--update', type=str)
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

    # validation
    validation = None
    if 'validation' in config:
        validation = {}
        val_img = Image.open(config['validation']['image_path'])
        validation['condition'] = utils.PIL2tensor(val_img)
        validation['postfix'] = os.path.splitext(config['validation']['image_path'])[-1]
        validation['output_dir'] = config['validation']['output_dir']
        utils.mkdir(validation['output_dir'])

    # palette
    model = Palette(config['model'], logger=logger)
    model.train(train_epochs=config['train']['epochs'], data_loader=train_data_loader, validation=validation)

    # inference
    utils.mkdir(config['test']['output_dir'])
    model.inference(data_loader=test_data_loader, output_dir=config['test']['output_dir'])