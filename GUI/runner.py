from PyQt6.QtCore import QObject, QThread, pyqtSignal
import time
import torch
from torchvision import transforms
from PIL import Image
from models.palette import Palette
import utils


class ModelRunner(QObject):
    # signals
    __init_model_done_signal__ = pyqtSignal()
    __load_checkpoint_done_signal__ = pyqtSignal()
    __inference_start_signal__ = pyqtSignal(str)
    __inference_done_signal__ = pyqtSignal()

    def __init__(self, config=None, logger=None):
        super(ModelRunner, self).__init__()
        self.config = config
        self.logger = logger
        self.model = None
        self.path_to_checkpoint = ''
        self.result = None
        self.sketch = None
        self.tf_reference = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])
        self.tf_condition = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])

    def init_model(self):
        if not self.config:
            return
        self.model = Palette(self.config['model'], logger=self.logger)
        if 'load_path' in self.config['model']['status'] and self.config['model']['status']['load_path']:
            self.path_to_checkpoint = self.config['model']['status']['load_path']
            self.__load_checkpoint_done_signal__.emit()
        self.__init_model_done_signal__.emit()

    def load_checkpoint(self, filename):
        self.model.load_status(filename)
        self.path_to_checkpoint = filename
        self.__load_checkpoint_done_signal__.emit()

    def inference(self, filename):
        reference = Image.open(filename).convert('RGB')
        ref_tsr = self.tf_reference(utils.warp_image(reference)).unsqueeze(0)
        skt_tsr = self.tf_condition(self.sketch).unsqueeze(0)
        self.__inference_start_signal__.emit('Infering...')
        time_s = time.time()
        x_con = torch.cat([skt_tsr, ref_tsr], dim=1).to(self.model.device)
        x_ret = self.model.inference(x_con=x_con)[-1]
        time_e = time.time()
        x_pil = utils.tensor2PIL(x_ret)
        self.result = x_pil[0]
        self.__inference_done_signal__.emit()
        self.logger.info(f'Inference finished, time cost: {time_e - time_s}s')
