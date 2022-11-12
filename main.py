import argparse
import json
import logging
import os
from torchvision import datasets, transforms
import utils
from models.palette import Palette


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    args = parser.parse_args()

    # logger initialize
    logging.basicConfig(format='[%(asctime)s][%(levelname)s] %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.DEBUG)

    try:
        with open(args.config, 'r') as fr:
            config = json.load(fr)
    except FileNotFoundError:
        logging.error("No such file: {}".format(args.config))
        exit()
    except PermissionError:
        logging.error("Unable to access: {}".format(args.config))
        exit()

    # folders
    for folder in config['directories'].values():
        if len(folder) == 0:
            continue
        mkdir_status = utils.mkdir(folder)
        if not mkdir_status:
            logging.error('Unable to create {0}'.format(folder))
            exit()

    # dataset
    transform = transforms.Compose([
        utils.Preprocess()
    ])
    train_dataset = datasets.CIFAR10(os.path.join(config['directories']['dataset'], 'cifar_10'), train=True, download=True, transform=transform)

    # palette
    model = Palette(config['model'])
    if 'status_load' in config['directories'] and os.access(config['directories']['status_load'], os.R_OK):
        model.load_status(config['directories']['status_load'])
    model.train(dataset=train_dataset, status_save_dir=config['directories']['status_save'])

    # inference
    test_dataset = datasets.CIFAR10(os.path.join(config['directories']['dataset'], 'cifar_10'), train=False, download=False, transform=transform)
    model.inference(test_dataset, config['directories']['test_result'])