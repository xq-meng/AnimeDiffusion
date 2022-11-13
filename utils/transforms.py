import numpy as np
import torch
import utils
from torchvision import transforms
from PIL import Image


class Preprocess(object):
    def __init__(self):
        pass

    def __call__(self, image):
        """
        :param[in]  image   PIL Image
        :return     torch.Tensor    [3 x h x w] (3 channel: L, a, b)
        """
        img_arr = np.array(image)
        h, w, _ = img_arr.shape
        for i in range(h):
            for j in range(w):
                img_arr[i][j] = utils.rgb2lab(img_arr[i][j][0], img_arr[i][j][1], img_arr[i][j][2])
        return transforms.ToTensor()(img_arr)


class Postprocess(object):
    def __init__(self):
        pass

    def __call__(self, image):
        """
        :param[in]  image   torch.Tensor    [3 x h x w] (3 channel: L, a, b)
        """
        img_arr = np.array(image).astype(np.float32)
        h, w, c = img_arr.shape
        for i in range(h):
            for j in range(w):
                img_arr[i][j] = utils.rgb2lab(img_arr[i][j][0], img_arr[i][j][1], img_arr[i][j][2])
        img_arr[:, :, 0] = img_arr[:, :, 0] / 100
        img_arr[:, :, 1] = (img_arr[:, :, 1] + 128) / 255
        img_arr[:, :, 2] = (img_arr[:, :, 2] + 128) / 255
        return torch.from_numpy(img_arr)