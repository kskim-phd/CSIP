# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 2021
@author: Hyun Bin Cho (hbcho@naver.com)

Reference
https://github.com/jongcye/Deep-Learning-COVID-19-on-CXR-using-Limited-Training-Data-Sets
"""

import header

# common
import torch
from torchvision.transforms import functional as TF
import random
import numpy as np

# dataset
from torch.utils.data import Dataset, DataLoader
import os
import glob
from PIL import Image

# add
import csv
import shutil


class MyInferenceClass(Dataset):

    def __init__(self, image_path):

#         image_path = header.dir_data_root + tag
        self.images = glob.glob(image_path + '/*.png')
        self.images.sort()
        self.data_len = len(self.images)
        self.ids = []


        for sample in self.images:
            self.ids.append(sample.replace(image_path, ''))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):

        # load and preprocessing
        images = self.get_original(index).astype('float32')
        img_name = self.images[index]
        original_image_size = np.asarray(images.shape)

        # resize
        images = np.asarray(Image.fromarray(images).resize((header.resize_width, header.resize_height)))

        return {'input': np.expand_dims(images, 0), 'ids': self.ids[index], 'im_size': original_image_size, 'filename': img_name}

    def get_original(self, index):

        # load image
        images = Image.open(self.images[index])
        if (np.asarray(images).max() <= 255):
            images = images.convert("L")
        images = np.asarray(images)

        # crop blank area - public only
        line_center = images[int(images.shape[0] / 2):, int(images.shape[1] / 2)]
        if (line_center.min() == 0):
            images = images[:int(images.shape[0] / 2) + np.where(line_center == 0)[0][0], :]

        # preprocessing
        images = pre_processing(images, flag_jsrt=0)

        return images


def pre_processing(images, flag_jsrt=10):
    # histogram
    num_out_bit = 1 << header.rescale_bit
    num_bin = images.max() + 1

    # histogram specification, gamma correction
    hist, bins = np.histogram(images.flatten(), num_bin, [0, num_bin])
    cdf = hist_specification(hist, num_out_bit, images.min(), num_bin, flag_jsrt)
    images = cdf[images].astype('float32')

    return images


def hist_specification(hist, bit_output, min_roi, max_roi, flag_jsrt):
    cdf = hist.cumsum()
    cdf = np.ma.masked_equal(cdf, 0)

    # hist sum of low & high
    hist_low = np.sum(hist[:min_roi + 1]) + flag_jsrt
    hist_high = cdf.max() - np.sum(hist[max_roi:])

    # cdf mask
    cdf_m = np.ma.masked_outside(cdf, hist_low, hist_high)

    # build cdf_modified
    if not (flag_jsrt):
        cdf_m = (cdf_m - cdf_m.min()) * (bit_output - 1) / (cdf_m.max() - cdf_m.min())
    else:
        cdf_m = (bit_output - 1) - (cdf_m - cdf_m.min()) * (bit_output - 1) / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m.astype('float32'), 0)

    # gamma correction
    cdf = pow(cdf / (bit_output - 1), header.gamma) * (bit_output - 1)

    return cdf


def one_hot(x, class_count):
    return torch.eye(class_count)[:, x]


def create_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_size_id(idx, size, case_id, dir_label):
    original_size_w_h = (size[idx][1].item(), size[idx][0].item())
    case_id = case_id[idx]
    dir_results = [case_id + case_id + '_' + j + '.png' for j in dir_label]

    return original_size_w_h, case_id, dir_results


def split_dataset(len_dataset):
    # set parameter
    offset_split_train = int(np.floor(header.train_split * len_dataset))
    offset_split_valid = int(np.floor(header.valid_split * len_dataset))
    indices = list(range(len_dataset))

    # shuffle
    np.random.seed(407)
    np.random.shuffle(indices)

    # set samplers
    train_sampler = indices[:offset_split_train]
    valid_sampler = indices[offset_split_train:offset_split_valid]
    test_sampler = indices[offset_split_train:]  # [offset_split_valid:]

    return train_sampler, valid_sampler, test_sampler