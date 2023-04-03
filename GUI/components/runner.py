from PyQt6.QtCore import QObject, pyqtSignal
import time
import torch
from torchvision import transforms
from PIL import Image
from base.model_handler import ModelHandler
from utils.image import warp_image, tensor2PIL, XDoG


class ModelRunner(QObject):
    # signals
    __init_model_done_signal__ = pyqtSignal()
    __load_checkpoint_done_signal__ = pyqtSignal()
    __inference_start_signal__ = pyqtSignal(str)
    __inference_done_signal__ = pyqtSignal()

    def __init__(
        self,
        model_option=None,
        logger=None,
        **kwargs
    ):
        super(ModelRunner, self).__init__()
        self.model_option = model_option
        self.logger = logger
        self.model_handler = None
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
        if not self.model_option:
            return
        self.model_handler = ModelHandler(**self.model_option)
        self.xdog = XDoG()
        self.device = self.model_handler.device
        self.__init_model_done_signal__.emit()

    def load_checkpoint(self, filename):
        self.model_handler.load_checkpoint(filename)
        self.path_to_checkpoint = filename
        self.__load_checkpoint_done_signal__.emit()

    def set_sketch(self, image):
        self.sketch = self.xdog(image.convert('RGB')).convert('L')

    def inference(self, filename):
        # input image
        reference = Image.open(filename).convert('RGB')
        # to tensor
        x_ref = self.tf_reference(warp_image(reference)).unsqueeze(0).to(self.device)
        x_con = self.tf_condition(self.sketch).unsqueeze(0).to(self.device)
        # denoising ddim
        self.__inference_start_signal__.emit('Infering...')
        time_s = time.time()
        with torch.no_grad():
            noise = torch.randn_like(x_ref).to(self.device)
            x_ret = self.model_handler.model.inference_ddim(x_t=noise, x_cond=torch.cat([x_con, x_ref], dim=1))[-1]
        time_e = time.time()
        # output result
        x_pil = tensor2PIL(x_ret)
        self.result = x_pil[0]
        self.__inference_done_signal__.emit()
        self.logger.info(f'Inference finished, time cost: {(time_e - time_s):.4f}s')