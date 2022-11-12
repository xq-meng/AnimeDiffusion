import numpy as np
import utils
from torchvision import transforms


class Preprocess(object):
    def __init__(self):
        pass

    def __call__(self, image):
        img_arr = np.array(image)
        h, w, _ = img_arr.shape
        for i in range(h):
            for j in range(w):
                img_arr[i][j] = utils.rgb2lab(img_arr[i][j][0], img_arr[i][j][1], img_arr[i][j][2])
        return transforms.ToTensor()(img_arr)
