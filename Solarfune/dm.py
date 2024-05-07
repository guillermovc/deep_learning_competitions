from pathlib import Path

import torch 
from skimage import io
import numpy as np
from torchvision.transforms import v2


class Dataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, trans):
        self.images = images
        self.labels = labels
        self.trans = trans

        self.imgs_path = Path("./data/train/")

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = io.imread(self.imgs_path / self.images[index])
        label = self.labels[index]
        # print(type(image))
        image = self.trans(image)
        # print(type(image))

        return image.float(), label.long()