from utils.image.thinplate.numpy import *

try:
    import torch
    import utils.image.thinplate.pytorch as torch
except ImportError:
    pass

__version__ = '1.0.0'