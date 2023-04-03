import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from utils.image import warp_image


class Anime(Dataset):

    def __init__(
        self,
        reference_path,
        condition_path,
        size=512,
        torch_dtype='float32'
    ):
        self.reference_path = reference_path
        self.condition_path = condition_path
        self.tf_ref = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])
        self.tf_con = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
        self.ref_filenames = os.listdir(self.reference_path)
        self.con_filenames = os.listdir(self.condition_path)
        self.ref_filenames.sort()
        self.con_filenames.sort()
        self.size = size
        self.dtype = getattr(torch, torch_dtype)

    def __getitem__(self, index):
        assert self.ref_filenames[index] == self.con_filenames[index]

        ret = {}
        img_reference = Image.open(os.path.join(self.reference_path, self.ref_filenames[index])).convert('RGB')
        img_condition = Image.open(os.path.join(self.condition_path, self.con_filenames[index])).convert('L')
        img_distorted = warp_image(img_reference)

        ret['reference'] = self.tf_ref(img_reference).to(self.dtype)
        ret['condition'] = self.tf_con(img_condition).to(self.dtype)
        ret['distorted'] = self.tf_ref(img_distorted).to(self.dtype)
        ret['name'] = self.con_filenames[index]
        return ret

    def __len__(self):
        return len(self.con_filenames)