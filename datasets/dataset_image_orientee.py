from PIL import Image
import numpy as np
import os
from os import path as osp
from pycocotools.coco import COCO
from config import CFG

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


import rasterio


class ImageOrienteeDatasetTest(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = [file for file in os.listdir(self.image_dir) if osp.isfile(osp.join(self.image_dir, file))]

    def __getitem__(self, index):
        img_path = osp.join(self.image_dir, self.images[index])
        image = rasterio.open(img_path).read()
        #image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform is not None:
            image = self.transform(image=image)['image']

        image = torch.FloatTensor(image)
        return image

    def __len__(self):
        return len(self.images)