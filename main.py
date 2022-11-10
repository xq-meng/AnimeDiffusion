import argparse
import json
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

    model = Palette(config['model'])