import torch.utils.data as data
from PIL import Image
import random
import time
import numpy as np
import os
import os.path as osp
from pdb import set_trace as st

class VisDaDATA(data.Dataset):
    def __init__(self, root, is_train=True, transform=None, shots=-1, seed=0, preload=False):
        self.transform = transform
        self.num_classes = 12
        train_dir = osp.join(root, "train")

        mapfile = os.path.join(train_dir, 'image_list.txt') 
        self.train_data_list = []
        self.test_data_list = []
        with open(mapfile) as f:
            for i, line in enumerate(f):
                path, class_idx = line.split()
                class_idx = int(class_idx)
                path = osp.join(train_dir, path)
                if i%10 == 0:
                    self.test_data_list.append((path, class_idx))
                else:
                    self.train_data_list.append((path, class_idx))

        if is_train:
            self.data_list = self.train_data_list
            if shots > 0:
                random.shuffle(self.train_data_list)
                self.data_list = self.train_data_list[:shots]
        else:
            self.data_list = self.test_data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        path, label = self.data_list[index]

        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label
        
