import sys
import argparse
import json
from PyQt6.QtWidgets import QApplication
from GUI.animeUI import AnimeUI


def UI(*args, **kwargs):
    app = QApplication(sys.argv)
    w = AnimeUI(*args, **kwargs)
    sys.exit(app.exec())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='')
    args = parser.parse_args()
    
    config = None
    if args.config:
        with open(args.config, 'r') as fr:
            config = json.load(fr)

    UI(config)