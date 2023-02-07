from PyQt6 import QtCore
from PyQt6.QtWidgets import QListView, QMenu
from PyQt6.QtGui import QStandardItem, QStandardItemModel, QPixmap, QIcon, QAction
from PyQt6.QtCore import QSize, pyqtSignal


class CandidateList(QListView):
    __emit_reference_signal__ = pyqtSignal(str)

    def __init__(self, item_width, item_height, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # context menu
        self.__init_context_menu__()
        # QListView
        self.item_model = QStandardItemModel()
        self.setModel(self.item_model)
        self.item_size = QSize(item_width, item_height)
        self.setIconSize(self.item_size)
        self.setGridSize(self.item_size)
        # data
        self.data = []
        # tmp varibles

    def __init_context_menu__(self):
        self.action_delete_item = QAction('Delete item', self)
        self.action_delete_item.triggered.connect(self.deleteCandidate)
        self.context_menu = QMenu(self)
        self.context_menu.addAction(self.action_delete_item)

    def mouseDoubleClickEvent(self, event):
        self.__emit_reference_signal__.emit(self.get())

    def contextMenuEvent(self, event):
        if not self.selectionModel().selectedIndexes():
            return
        self.context_menu.exec(event.globalPos())

    def appendCandidate(self, item: QPixmap, value):
        item = item.scaled(self.item_size, transformMode=QtCore.Qt.TransformationMode.SmoothTransformation)
        self.data.append(value)
        self.item_model.appendRow(QStandardItem(QIcon(item), ''))

    def deleteCandidate(self):
        indices = self.selectionModel().selectedIndexes()
        indices.sort()
        for x in reversed(indices):
            self.data.pop(x.row())
            self.item_model.removeRow(x.row())

    def get(self):
        if not self.selectionModel().selectedIndexes():
            return ''
        index = self.selectionModel().selectedIndexes()[0].row()
        return self.data[index]
