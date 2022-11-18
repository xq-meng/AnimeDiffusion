import math
import random
import numpy as np
import cv2
from PIL import Image
import utils.thinplate as tps

def warp_image(image):
    """
    :param[in]  image           type: PIL Image
    :return     distorted image type: PIL Image
    """
    p1, r1, p2, r2 = 0.0, 0.0, 0.0, 0.0
    while math.isclose(p1 + r1, p2 + r2):
        p1 = round(random.uniform(0.3, 0.7), 2)
        p2 = round(random.uniform(0.3, 0.7), 2)
        r1 = round(random.uniform(-0.25, 0.25), 2)
        r2 = round(random.uniform(-0.25, 0.25), 2)

    c_src = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [p1, p1], [p2, p2]])
    c_dst = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [p1 + r1, p1 + r1], [p2 + r2, p2 + r2]])
    arr = np.array(image)
    theta = tps.tps_theta_from_points(c_src, c_dst, reduced=True)
    grids = tps.tps_grid(theta, c_dst, arr.shape)
    mapx, mapy = tps.tps_grid_to_remap(grids, arr.shape)
    warped_arr = cv2.remap(arr, mapx, mapy, cv2.INTER_CUBIC)
    return Image.fromarray(warped_arr).rotate(random.randrange(360)).resize(image.size)