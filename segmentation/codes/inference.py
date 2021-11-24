# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 2021
@author: Hyun Bin Cho (hbcho@naver.com)

Reference
https://github.com/jongcye/Deep-Learning-COVID-19-on-CXR-using-Limited-Training-Data-Sets
"""

import header

# common
import torch, torchvision
import numpy as np

# dataset
import mydataset
from torch.utils.data import DataLoader
from PIL import Image

# model
import model
import torch.optim as optim
import torch.nn as nn
import os
import glob

# post processing
import cv2

from tqdm import tqdm
import random
import shutil

def post_processing(raw_image, original_size, flag_pseudo=0):

    net_input_size = raw_image.shape
    raw_image = raw_image.astype('uint8')

    # resize
    if (flag_pseudo):
        raw_image = cv2.resize(raw_image, original_size, interpolation=cv2.INTER_NEAREST)
    else:
        raw_image = cv2.resize(raw_image, original_size, interpolation=cv2.INTER_NEAREST)    

    if (flag_pseudo):
        raw_image = cv2.resize(raw_image, net_input_size, interpolation=cv2.INTER_NEAREST)

    return raw_image

print("\ninference.py")

##############################################################################################################################
# Semantic segmentation (inference)

# Flag  
flag_eval_JI = False #False # calculate JI
flag_save_JPG = True # preprocessed, mask

# GPU   
if torch.cuda.is_available():
    device = torch.device("cuda:0") 
    num_worker = header.num_worker 
else:
    device = torch.device("cpu") 
    num_worker = 0


# Model initialization
net = header.net


# Load model
model_dir = header.dir_checkpoint + header.filename_model
if os.path.isfile(model_dir):
    print('\n>> Load model - %s' % (model_dir))
    checkpoint = torch.load(model_dir)
    net.load_state_dict(checkpoint['model_state_dict']) 
    test_sampler = checkpoint['test_sampler']
    print("  >>> Epoch : %d" % (checkpoint['epoch']))
    # print("  >>> JI Best : %.3f" % (checkpoint['ji_best']))
else:
    print('[Err] Model does not exist in %s' % (header.dir_checkpoint + header.filename_model))
    exit()

# network to GPU
net.to(device) 

# loop dataset class #alldata


print('\n>> Load dataset ')
testset = mydataset.MyInferenceClass(image_path = header.datadir)
testloader = DataLoader(testset, batch_size=header.num_batch_test, shuffle=False, num_workers=num_worker, pin_memory=True)
print("  >>> Total # of test sampler : %d" % (len(testset)))
# inference
print('\n\n>> Evaluate Network')
with torch.no_grad():
    # initialize
    net.eval()
    ji_test = []
    for i, data in enumerate(testloader, 0):
        # forward
        outputs = net(data['input'].to(device))
        outputs = torch.argmax(outputs.detach(), dim=1)

        # one hot
        outputs_max = torch.stack([mydataset.one_hot(outputs[k], header.num_masks) for k in range(len(data['input']))])

        # each case
        for k in range(len(data['input'])):

            # get size and case id
            original_size, dir_case_id, dir_results = mydataset.get_size_id(k, data['im_size'], data['ids'], header.net_label[1:])
            # post processing
            # if os.path.exists(save_dir + dir_case_id + '_image.png'):
            #     continue
            post_output = [post_processing(outputs_max[k][j].numpy(), original_size) for j in range(1, header.num_masks)] # exclude background

            # original image processings


            image_original = testset.get_original(i*header.num_batch_test+k)

            X_whole = np.asarray(image_original)
            h_whole = X_whole.shape[0] # original w
            w_whole = X_whole.shape[1] # original h
#
            Image.fromarray(image_original.astype('uint8')).convert('L').save(data['filename'][k])
            Image.fromarray(post_output[1]*255 + post_output[2]*255).convert('L').save(data['filename'][k]+ '_mask.jpg')

print('\n\n>> Done')