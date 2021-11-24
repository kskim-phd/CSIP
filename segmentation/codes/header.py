# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 2021
@author: Hyun Bin Cho (hbcho@naver.com)

Reference
https://github.com/jongcye/Deep-Learning-COVID-19-on-CXR-using-Limited-Training-Data-Sets
"""

import model
import torch


# Model
tag = 'segmentation'
filename_model = 'model_' + tag + '.pth'
dataset = 0
division_trainset = 1
threshold_partial = 1
partial_dataset = False
gamma = 0.5
num_channel = 1
ratio_dropout = 0.2
weight_bk = 0.5

# Directory 

datadir = '../../inputs/*/*'
dir_checkpoint = "../segmentation_checkpoint/"


# Network
num_masks = 4
num_network = 1
net = model.FCDenseNet(num_channel, num_masks, ratio_dropout) 
net_label = ['BG', 'Cardiac', 'Thorax(L)', 'Thorax(R)']

# Dataset
orig_height = 2048
orig_width = 2048
resize_height = 256 
resize_width = 256 
rescale_bit = 8 

# CPU
num_worker = 0


# Test schedule
num_batch_test = 2

