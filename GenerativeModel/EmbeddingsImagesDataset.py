# coding=utf-8


"""
Author: zhangjing
Date and time: 2/02/19 - 17:58

Revised by QiuXiang
date: 03/17/21
"""

import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from config.config import net_config
from utils import get_nb_files


# This class is to generate self-defined datasets, and hence we can use "dataloader" function in pytorch
# note that our EmbeddingsImagesDataset in inherited from "Dataset" class in "torch.utils.data"
class EmbeddingsImagesDataset(Dataset):
    def __init__(self, dir_z, dir_x, nb_channels=3):

        # check the number of files in X(means original image) and Z(the implicit vector after vector) are equal?
        assert get_nb_files(dir_z) == get_nb_files(dir_x), \
            "The files numbers are not equal. \n Please check infomation in {0} and {1}\n".format(dir_z, dir_x)
        assert nb_channels in [1, 3]

        self.nb_files = get_nb_files(dir_z)

        self.nb_channels = nb_channels

        self.dir_z = dir_z
        self.dir_x = dir_x
        self.filename = os.listdir(dir_x)

    def __len__(self):
        return self.nb_files

    def __getitem__(self, idx):
        # filename = os.path.join(self.dir_z, '{}.npy'.format(idx))
        filename = os.path.join(self.dir_z, self.filename[idx].split('.')[0]) + '.npy'
        z = np.load(filename)

        # filename = os.path.join(self.dir_x, '{}.jpg'.format(idx))
        filename = os.path.join(self.dir_x, self.filename[idx].split('.')[0]) + '.jpg'
        if self.nb_channels == 3:
            if (net_config['train_x'] == '(-1,1)'):
                x = (np.ascontiguousarray(Image.open(filename), dtype=np.uint8).transpose((2, 0, 1)) / 127.5) - 1.0
            elif (net_config['train_x'] == '(0,1)'):
                x = (np.ascontiguousarray(Image.open(filename), dtype=np.uint8).transpose((2, 0, 1)) / 255)
            else:
                print("训练时x输入ew范围配置出错...请检查")
                exit(0)
        else:
            x = np.expand_dims(np.ascontiguousarray(Image.open(filename), dtype=np.uint8), axis=-1)
            if (net_config['train_x'] == '(-1,1)'):
                x = (x.transpose((2, 0, 1)) / 127.5) - 1.0
            elif (net_config['train_x'] == '(0,1)'):
                x = (x.transpose((2, 0, 1)) / 255)
            else:
                print("训练时x输入范围配置出错...请检查")
                exit(0)

        sample = {'z': z, 'x': x}
        return sample
