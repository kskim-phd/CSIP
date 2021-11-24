"""
Created on Mon Aug 19 2021
@author: Hyun Bin Cho (hbcho@naver.com)

Reference
https://github.com/alinlab/CSI
"""

import os
import numpy as np
import torch
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms
from utils.utils import set_random_seed
from torch.utils import data
import glob
from PIL import Image
import random
import torchvision.transforms.functional as functional
from sklearn.utils import resample

def get_dataset(P, image_size=None):
    image_size = (P.cxr_crop, P.cxr_crop, 3)
    n_classes = 2
    test_set = CXR_Dataset(mode='test', resize = (P.cxr_resize, P.cxr_resize), crop_size=(P.cxr_crop, P.cxr_crop), P=P)
    return  test_set, image_size, n_classes

def get_subclass_dataset(dataset, classes):
    if not isinstance(classes, list):
        classes = [classes]

    indices = []
    for idx, tgt in enumerate(dataset.targets):
        if tgt in classes:
            indices.append(idx)

    dataset = Subset(dataset, indices)
    return dataset


def augmentation(image, rand_p, mode):
    if mode == 'train':
        # random vertical flip
        if rand_p > 0.5:
            image = functional.hflip(image)
        else:
            pass
    elif mode == 'test':
        pass
    else:
        print('Error: not a valid phase option.')

    return image

class CXR_Dataset(data.Dataset):
    def __init__(self, mode, resize, crop_size, P):
        self.mode = mode
        self.resize = resize
        self.crop_size = crop_size
        data_dir = os.path.join(P.input_dir, self.mode)
        labels = os.listdir(data_dir)
        self.total_images_dic = {}
        self.targets = []
        for label in labels:
            png_dir = os.path.join(data_dir, label)
            images_list = glob.glob(png_dir + '/*.png')
            for image in images_list:
                self.total_images_dic[image] = label
                self.targets.append(int(label))

        self.patch_data_transform = transforms.Compose([
                            transforms.Resize(crop_size),
                            transforms.ToTensor(),
        ])
        
        self.global_data_transform = transforms.Compose([
                            transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                            transforms.ToTensor(),
        ])
        self.P = P
    def __len__(self):
        return len(self.total_images_dic)    
    
    def generate_patch(self, whole_img, whole_mask):
        rand_p = random.random()
        h_whole = whole_img.shape[0] # original w
        w_whole = whole_img.shape[1] # original h
        whole_mask = np.round((whole_mask)/255)
        non_zero_list = np.nonzero(whole_mask)
        non_zero_row = random.choice(non_zero_list[0]) # random non-zero row index
        non_zero_col = random.choice(non_zero_list[1]) # random non-zero col index
        X_patch = whole_img[int(max(0, non_zero_row - (self.crop_size[0] / 2))):
                           int(min(h_whole, non_zero_row + (self.crop_size[0] / 2))),
                  int(max(0, non_zero_col - (self.crop_size[0] / 2))):
                  int(min(w_whole, non_zero_col + (self.crop_size[0] / 2)))]
        X_patch_img = self.patch_data_transform(augmentation(Image.fromarray(X_patch), rand_p=rand_p, mode=self.mode))
        X_patch_img_ = np.squeeze(np.asarray(X_patch_img))
        X_patch_1 = np.expand_dims(X_patch_img_, axis=0)
        X_patch_2 = np.expand_dims(X_patch_img_, axis=0)
        X_patch_3 = np.expand_dims(X_patch_img_, axis=0)
        X_ = np.concatenate((X_patch_1, X_patch_2, X_patch_3), axis=0)
        X = torch.from_numpy(X_)
        return X

    def __getitem__(self, index):
        y = list(self.total_images_dic.values())[index]
        img_dir = list(self.total_images_dic.keys())[index].replace("\\", "/")
        X_whole = np.asarray(Image.open(img_dir).resize(self.resize))
        X_whole_mask = np.array(Image.open(img_dir+'_mask.jpg').resize(self.resize))
        X = self.generate_patch(X_whole, X_whole_mask)
        data = {'img': X, 'label': int(y), 'img_dir': img_dir}
        return data