import argparse
import json
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import utils
from models.palette import Palette


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as fr:
            config = json.load(fr)
    except FileNotFoundError:
        print("No such file: {}".format(args.config))
    except PermissionError:
        print("Unable to access: {}".format(args.config))

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