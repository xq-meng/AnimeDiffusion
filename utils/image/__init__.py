from utils.image.wrap import warp_image
from utils.image.tensor2PIL import tensor2PIL
from utils.image.CIELab import ToCIELab, CIELabToRGB
from utils.image.xdog import XDoG


def is_image(filename: str):
    IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif', '.TIF', '.tiff', '.TIFF']
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)