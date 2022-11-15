import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import utils


class ColorizationDataset(Dataset):

    def __init__(self, reference_path, condition_path):
        self.reference_path = reference_path
        self.condition_path = condition_path
        self.tf_reference = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])
        self.tf_condition = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
        self.filenames = []
        for filename in os.listdir(self.reference_path):
            if utils.is_image(filename) and os.access(os.path.join(self.reference_path, filename), os.R_OK):
                self.filenames.append(filename)

    def __getitem__(self, index):
        ret = {}
        filename = self.filenames[index]
        img_reference = Image.open(os.path.join(self.reference_path, filename))
        img_condition = Image.open(os.path.join(self.condition_path, filename))
        ret['reference'] = self.tf_reference(img_reference)
        ret['condition'] = self.tf_condition(img_condition)
        ret['name'] = filename
        return ret

    def __len__(self):
        return len(self.filenames)