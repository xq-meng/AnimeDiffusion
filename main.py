import argparse
import json
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
    train_dataset = datasets.CIFAR10('./dataset/cifar_10', train=True, download=True, transform=transform)

    # palette
    model = Palette(config['model'])
    model.train(dataset=train_dataset)

    # inference
    test_dataset = datasets.CIFAR10('./dataset/cifar_10', train=False, download=False, transform=transform)
    model.inference(test_dataset, './out/test_results')