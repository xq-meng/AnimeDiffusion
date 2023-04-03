import os
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QGroupBox, QTextBrowser, QListView, QPushButton, QFileDialog
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtGui import QAction
from PyQt6.QtCore import QThread, QSize, QPoint, pyqtSignal
from PIL import Image, ImageQt
from GUI.components.message_box import MessageBox
from GUI.components.candidate_list import CandidateList
from GUI.components.runner import ModelRunner
from utils.logger import Logger


class AnimeUI(QMainWindow):
    # signals
    __init_model_start_signal__ = pyqtSignal(str)
    __load_sketch_done_signal__ = pyqtSignal()

    def __init__(self, option):
        super().__init__()
        self.__init_ui__()
        self.show()
        self.__init_data__(option=option)
        self.__init_connect__()
        self.__init_worker__()

    def __init_data__(self, option):
        # initial variables
        self.img_ref = []
        self.sketch_filename = ''
        self.sketch_image = None
        self.last_checkpoint_dir = ''
        self.last_reference_dir = ''
        self.last_sketch_dir = ''
        # logger
        ui_logger_handler = MessageBox(self.log_browser)
        ui_logger = Logger(name='ui', console=False, handler=ui_logger_handler)
        # distributed
        self.runner = ModelRunner(**option, logger=ui_logger)
        self.thread = QThread(self)
        self.runner.moveToThread(self.thread)

    def __init_connect__(self):
        # actions
        self.exit_action.triggered.connect(QApplication.instance().quit)
        self.load_checkpoint_action.triggered.connect(self.__open_checkpoint_file__)
        self.load_sketch_action.triggered.connect(self.__open_condition_image_file__)
        # buttons
        self.reference_add_button.clicked.connect(self.__open_reference_image_file__)
        # signals
        self.__init_model_start_signal__.connect(self.runner.init_model)
        self.__init_model_start_signal__.connect(self.__update_status_bar__)
        self.__load_sketch_done_signal__.connect(self.__print_sketch_path__)
        # components signals
        self.reference_candidate.__emit_reference_signal__.connect(self.runner.inference)
        self.runner.__load_checkpoint_done_signal__.connect(self.__print_checkpoint_path__)
        self.runner.__load_checkpoint_done_signal__.connect(self.__update_status_bar__)
        self.runner.__inference_start_signal__.connect(self.__update_status_bar__)
        self.runner.__inference_done_signal__.connect(self.__show_result__)
        self.runner.__inference_done_signal__.connect(self.__update_status_bar__)

    def __init_worker__(self):
        self.thread.start()
        self.__init_model_start_signal__.emit('Init model...')

    def __init_ui__(self):
        # window size.
        self.setWindowTitle('AnimeDiffusion')
        # actions
        self.exit_action = QAction('Exit', self)
        self.exit_action.setShortcut('Ctrl+Q')
        self.load_checkpoint_action = QAction('Load checkpoint', self)
        self.load_sketch_action = QAction('Load line drawing', self)
        # menu bar.
        file_menu = self.menuBar().addMenu('File')
        file_menu.addAction(self.load_checkpoint_action)
        file_menu.addAction(self.load_sketch_action)
        file_menu.addAction(self.exit_action)
        help_menu = self.menuBar().addMenu('Help')
        # status bar.
        self.statusBar().showMessage('Ready.')
        # load basic
        self.load_checkpoint_label = QLabel('Checkpoint: ', self)
        self.load_checkpoint_label.setMaximumWidth(100)
        self.load_sketch_label = QLabel('Line drawing: ', self)
        self.load_sketch_label.setMaximumWidth(100)
        self.load_checkpoint_path_label = QLabel('No checkpoint loaded.', self)
        self.load_sketch_path_label = QLabel('No line drawing loaded.', self)
        load_basic_box = QGroupBox('')
        load_basic_box_layout = QGridLayout()
        load_basic_box_layout.addWidget(self.load_checkpoint_label, 0, 0)
        load_basic_box_layout.addWidget(self.load_sketch_label, 1, 0)
        load_basic_box_layout.addWidget(self.load_checkpoint_path_label, 0, 1)
        load_basic_box_layout.addWidget(self.load_sketch_path_label, 1, 1)
        load_basic_box.setLayout(load_basic_box_layout)
        # sketch display.
        self.sketch_canvas = QLabel(self)
        self.sketch_canvas_size = QSize(300, 300)
        self.sketch_canvas.resize(self.sketch_canvas_size)
        self.sketch_canvas.setScaledContents(True)
        sketch_canvas_box = QGroupBox('Line drawing')
        sketch_canvas_box_layout = QHBoxLayout()
        sketch_canvas_box_layout.addWidget(self.sketch_canvas)
        sketch_canvas_box.setLayout(sketch_canvas_box_layout)
        # result displace
        self.result_canvas = QLabel(self)
        self.result_canvas_size = QSize(300, 300)
        self.result_canvas.resize(self.result_canvas_size)
        self.result_canvas.setScaledContents(True)
        result_canvas_box = QGroupBox('Colorization result')
        result_canvas_box_layout = QHBoxLayout()
        result_canvas_box_layout.addWidget(self.result_canvas)
        result_canvas_box.setLayout(result_canvas_box_layout)
        # main display area.
        display_layout = QHBoxLayout()
        display_layout.addWidget(sketch_canvas_box)
        display_layout.addWidget(result_canvas_box)
        display_layout.addStretch(1)
        # config pane area.
        config_pane_layout = QGridLayout()
        # message box.
        self.log_browser = QTextBrowser(self)
        self.log_browser.setMaximumHeight(100)
        log_browser_box = QGroupBox('Message box')
        log_browser_box_layout = QHBoxLayout()
        log_browser_box_layout.addWidget(self.log_browser)
        log_browser_box.setLayout(log_browser_box_layout)
        # main operation area.
        operation_layout = QVBoxLayout()
        operation_layout.addLayout(display_layout)
        operation_layout.addLayout(config_pane_layout)
        operation_layout.addStretch(1)
        # reference candidate
        self.reference_add_button = QPushButton('Add', self)
        self.reference_candidate = CandidateList(item_height=90, item_width=90, parent=self)
        self.reference_candidate.setFlow(QListView.Flow.TopToBottom)
        self.reference_candidate.setMaximumWidth(100)
        reference_candidate_box = QGroupBox('Reference')
        reference_candidate_box_layout = QVBoxLayout()
        reference_candidate_box_layout.addWidget(self.reference_add_button)
        reference_candidate_box_layout.addWidget(self.reference_candidate)
        reference_candidate_box.setLayout(reference_candidate_box_layout)
        # functional layout
        function_layout = QHBoxLayout()
        function_layout.addLayout(operation_layout)
        function_layout.addWidget(reference_candidate_box)
        # main layout.
        main_layout = QVBoxLayout()
        main_layout.addWidget(load_basic_box)
        main_layout.addLayout(function_layout)
        main_layout.addWidget(log_browser_box)
        main_layout.addStretch(1)
        # central widgets
        central_weight = QWidget()
        central_weight.setLayout(main_layout)
        self.setCentralWidget(central_weight)

    def __screen_center__(self):
        screen = QGuiApplication.primaryScreen().geometry()
        window = self.geometry()
        self.move(QPoint(screen.left() + int((screen.width() - window.width()) / 2), screen.top() + int((screen.height() - window.height()) / 2)))

    def __show_sketch__(self, img: Image.Image):
        img = img.resize((self.sketch_canvas_size.width(), self.sketch_canvas_size.height()), Image.ANTIALIAS).convert('RGBA')
        image = ImageQt.toqpixmap(img)
        self.sketch_canvas.setPixmap(image)

    def __show_result__(self, img: Image.Image=None):
        if img is None:
            img = self.runner.result
        img = img.resize((self.result_canvas_size.width(), self.result_canvas_size.height()), Image.ANTIALIAS).convert('RGBA')
        image = ImageQt.toqpixmap(img)
        self.result_canvas.setPixmap(image)

    def __print_checkpoint_path__(self):
        if self.runner.path_to_checkpoint:
            self.load_checkpoint_path_label.setText(self.runner.path_to_checkpoint)

    def __print_sketch_path__(self):
        if self.sketch_filename:
            self.load_sketch_path_label.setText(self.sketch_filename)

    def __append_reference_candidate__(self, filename):
        image = ImageQt.toqpixmap(Image.open(filename).convert('RGB'))
        self.reference_candidate.appendCandidate(image, value=filename)

    def __open_checkpoint_file__(self):
        dir = self.last_checkpoint_dir if self.last_checkpoint_dir else os.getcwd()
        filename, _ = QFileDialog.getOpenFileName(self, 'Select checkpoint', dir, 'Checkpoint Files (*.pkl)')
        if filename:
            self.last_checkpoint_dir, _ = os.path.split(filename)
            self.runner.load_checkpoint(filename)

    def __open_condition_image_file__(self):
        dir = self.last_sketch_dir if self.last_sketch_dir else os.getcwd()
        filename, _ = QFileDialog.getOpenFileName(self, 'Select image', dir, 'JPEG Files (*.jpg);;PNG Files (*.png)')
        if filename:
            self.sketch_image = Image.open(filename)
            self.sketch_filename = filename
            self.runner.set_sketch(self.sketch_image)
            self.__show_sketch__(self.sketch_image)
            self.last_sketch_dir, _ = os.path.split(filename)
            self.__load_sketch_done_signal__.emit()

    def __open_reference_image_file__(self):
        dir = self.last_reference_dir if self.last_reference_dir else os.getcwd()
        filename, _ = QFileDialog.getOpenFileName(self, 'Select image', dir, 'JPEG Files (*.jpg);;PNG Files (*.png)')
        if filename:
            self.__append_reference_candidate__(filename=filename)
            self.last_reference_dir, _ = os.path.split(filename)

    def __update_status_bar__(self, message=None):
        if not message:
            message = 'Ready.'
        self.statusBar().showMessage(message)

    def show(self):
        self.__show_sketch__(Image.new(mode='RGBA', size=(self.sketch_canvas_size.width(), self.sketch_canvas_size.height())))
        self.__show_result__(Image.new(mode='RGBA', size=(self.result_canvas_size.width(), self.result_canvas_size.height())))
        # self.__screen_center__()
        super().show()

    def closeEvent(self, event) -> None:
        self.thread.quit()
        self.thread.wait()
        return super().closeEvent(event)
