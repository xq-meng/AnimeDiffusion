import numpy as np
import cv2
from PIL import Image


class XDoG:
    def __init__(self,
                 k_sigma=3.5,
                 p=10,
                 epsilon=0.01,
                 phi=10**9):
        self.sigma_range = [0.3, 0.4, 0.5]
        self.k_sigma = k_sigma
        self.p = p
        self.epsilon = epsilon
        self.phi = phi

    @staticmethod
    def _sharpimage(img: np.array,
                    sigma: float,
                    k_sigma: float,
                    p: float) -> np.array:

        sigma_large = sigma * k_sigma
        G_small = cv2.GaussianBlur(img, (0, 0), sigma)
        G_large = cv2.GaussianBlur(img, (0, 0), sigma_large)
        S = (1+p) * G_small - p * G_large

        return S

    @staticmethod
    def _softthreshold(si: np.array,
                       epsilon: float,
                       phi: float) -> np.array:

        T = np.zeros(si.shape)
        si_bright = si >= epsilon
        si_dark = si < epsilon
        T[si_bright] = 1.0
        T[si_dark] = 1.0 + np.tanh(phi * (si[si_dark] - epsilon))

        return T

    def _xdog(self, img: np.array, sigma: float) -> np.array:
        s = self._sharpimage(img,
                             sigma,
                             self.k_sigma,
                             self.p)
        si = np.multiply(img, s)
        t = self._softthreshold(si, self.epsilon, self.phi)

        return t

    def __call__(self, image):
        img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
        img = img / 255.0
        sigma = np.random.choice(self.sigma_range)
        img = np.rint(self._xdog(img, sigma) * 255.0)
        return Image.fromarray(img)