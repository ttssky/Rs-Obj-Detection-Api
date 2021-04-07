
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

import albumentations as albu

from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from glob import glob

import torch
import numpy as np
from tqdm import tqdm
from shapely import wkb

class Dataset(BaseDataset):
    def __init__(
            self, 
            images_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)

        self.images_fps = sorted(glob(images_dir + '/*.png'))                
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # 读取
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 获取文件名
        name = os.path.basename(self.images_fps[i])
        
        # 预处理
        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample['image']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']
            
        return name,image
        
    def __len__(self):
        return len(self.ids)



def get_validation_augmentation():
    """验证集数据增强"""
    test_transform = [
        albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def query_image_meta(idataset, bbox):

    out = [el for el in idataset.query.all()]
    all_w_h = [(el.width, el.height) for el in out]
    all_bounds = [wkb.loads(str(el.geom), hex=True).bounds for el in out]
    for w_h,bounds in zip(all_w_h,all_bounds):
        x_min, y_min, x_max, y_max = bounds
        if (x_min <= bbox[0] and y_min <= bbox[1]) or \
            (x_max >= bbox[2] and y_max >= bbox[3]):
            w = w_h[0]
            h = w_h[1]
            sp_res = (((bounds[2] - bounds[0]) / w) + \
                ((bounds[3] - bounds[1]) / h)) / 2
            return w_h[0], w_h[1], sp_res
    else:
        raise ValueError


