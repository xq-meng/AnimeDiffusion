import os
import argparse
import json
import torch
from torch.utils.data import DataLoader
from PIL import Image
import utils
from models.palette import Palette
import datasets


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-x', '--update', type=json.loads, default=None)
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

    # update config with command line input
    utils.update_config(config=config, enhance_parse=args.update)

    # logger initialize
    logger = utils.Logger(name='base_logger', **config['logging'])

    # dataset
    distortion_guidance = config['components']['distortion_guidance']
    train_dataset = datasets.ColorizationDataset(**config['dataset']['train']['path'], distortion_guidance=distortion_guidance)
    train_data_loader = DataLoader(dataset=train_dataset ,**config['dataset']['train']['dataloader'])
    test_dataset = datasets.ColorizationDataset(**config['dataset']['test']['path'], distortion_guidance=distortion_guidance)
    test_data_loader = DataLoader(dataset=test_dataset, **config['dataset']['test']['dataloader'])

    # validation
    validations = []
    if 'validation' in config:
        for validation_config in config['validations']:
            validation = {}
            val_img = Image.open(validation_config['image_path'])
            validation['condition'] = utils.PIL2tensor(val_img)
            if distortion_guidance:
                val_distortion = Image.open(validation_config['distortion_guidance'])
                validation['condition'] = torch.cat([validation['condition'], utils.PIL2tensor(val_distortion)], dim=1)
            (validation['filename'], validation['postfix']) = os.path.splitext(validation_config['image_path'])
            validation['output_dir'] = validation_config['output_dir']
            validations.append(validation)
            utils.mkdir(validation['output_dir'])

    # palette
    model = Palette(config['model'], logger=logger)
    model.train(train_epochs=config['train']['epochs'], data_loader=train_data_loader, validations=validations)

    # inference
    utils.mkdir(config['test']['output_dir'])
    results = model.inference(data_loader=test_data_loader, output_dir=config['test']['output_dir'])