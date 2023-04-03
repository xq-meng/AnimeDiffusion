import sys
from GUI.animeUI import AnimeUI
from PyQt6.QtWidgets import QApplication


class GUI_ADF(object):
    def __init__(self, **kwargs):
        self.option = kwargs

    def run(self):
        app = QApplication(sys.argv)
        w = AnimeUI(self.option)
        sys.exit(app.exec())
