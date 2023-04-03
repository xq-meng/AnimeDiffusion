import torch
from torchvision.transforms.functional import to_tensor


class ToCIELab(object):
    def __init__(self, normalize=False) -> None:
        self.f1 = torch.tensor([[ 0.4887180, 0.1762044, 0.0000000], 
                                [ 0.3106803, 0.8129847, 0.0102048], 
                                [ 0.2006017, 0.0108109, 0.9897952]],
                               dtype=torch.float32)
        self.f2 = torch.tensor([[    0,  500,    0],
                                [  116, -500,  200],
                                [    0,    0, -200]],
                               dtype=torch.float32)
        self.normalize = normalize

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor in CIELab.

        Returns:
            Tensor: Converted image.
        """
        rgb_tsr = to_tensor(pic=pic).permute(1, 2, 0)
        forward_gamma_seg = rgb_tsr > 0.04045
        rgb_tsr = torch.pow((rgb_tsr + 0.055) / 1.055, 2.4) * forward_gamma_seg + (rgb_tsr / 12.92) * ~forward_gamma_seg
        xyz_tsr = torch.matmul(rgb_tsr, self.f1)
        forward_theta_seg = xyz_tsr > 0.008856
        xyz_tsr = torch.pow(xyz_tsr, 0.3333333) * forward_theta_seg + (xyz_tsr * 7.787 + 0.1379310) * ~forward_theta_seg
        lab_tsr = torch.matmul(xyz_tsr, self.f2)
        lab_tsr[:, :, 0] = torch.clamp(lab_tsr[:, :, 0] - 16.0, min=0., max=100.)
        lab_tsr[:, :, 1:] = torch.clamp(lab_tsr[:, :, 1:], min=-128., max=127.)
        if self.normalize:
            lab_tsr[:, :, 0] = lab_tsr[:, :, 0] / 100.
            lab_tsr[:, :, 1:] = (lab_tsr[:, :, 1:] + 128.) / 255.
        return lab_tsr.permute(2, 0, 1)


class CIELabToRGB(object):
    def __init__(self) -> None:
        self.f2 = torch.tensor([[ 1./116, 1./116, 1./116],
                                [  0.002,      0,      0],
                                [      0,      0, -0.005]],
                               dtype=torch.float32)
        self.f1 = torch.tensor([[ 2.3706743,-0.5138850, 0.0052982],
                                [-0.9000405, 1.4253036,-0.0146949],
                                [-0.4706338, 0.0885814, 1.0093968]],
                               dtype=torch.float32)

    def __call__(self, lab_tsr):
        """
        Args:
            lab_tsr (torch.Tensor): Tensor in CIELab, L \in [0, 100], a, b \in [-128, 127].

        Returns:
            Tensor: Converted tensor [0, 1].
        """
        lab_tsr = torch.clone(lab_tsr).permute(1, 2, 0)
        lab_tsr[:, :, 0] = lab_tsr[:, :, 0] + 16.0
        xyz_tsr = torch.matmul(lab_tsr, self.f2)
        reverse_theta_seg = xyz_tsr > 0.206892672
        xyz_tsr = torch.pow(xyz_tsr, 3) * reverse_theta_seg + ((xyz_tsr - 0.1379310) / 7.787) * ~reverse_theta_seg
        rgb_tsr = torch.matmul(xyz_tsr, self.f1)
        reverse_gamma_seg = rgb_tsr > 0.003130805
        rgb_tsr = (torch.pow(rgb_tsr, 0.416666667) * 1.055 - 0.055) * reverse_gamma_seg + (rgb_tsr * 12.92) * ~reverse_gamma_seg
        return rgb_tsr.permute(2, 0, 1)