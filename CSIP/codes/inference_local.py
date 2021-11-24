"""
Created on Mon Aug 19 2021
@author: Hyun Bin Cho (hbcho@naver.com)

Reference
https://github.com/alinlab/CSI
"""

from copy import deepcopy
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils import data
import os
import glob
import numpy as np
import random
from datasets.datasets import *
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from models.resnet_imagenet import resnet18
from utils.utils import set_random_seed, normalize
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import csv
matplotlib.use('Agg')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from argparse import ArgumentParser


def parse_args(default=False):
    """Command-line argument parser for training."""

    parser = ArgumentParser(description='Pytorch implementation of CSI')

    parser.add_argument('--train_iteration', help='Number of trarining iteration', default="1", type=int)
    parser.add_argument('--val_iteration', help='Number of validation iteration', default="5", type=int)
    parser.add_argument('--mode', choices=['simclr', 'simclr_CSI'], default="simclr_CSI", type=str)
    parser.add_argument('--num_worker', help='number of workers', default=2, type=int)


    parser.add_argument('--cxr_resize', help='Resize for CXR data', default=512, type=int)
    parser.add_argument('--cxr_crop', help='Random cropping for CXR data', default=64, type=int)
    parser.add_argument('--simclr_dim', help='Dimension of simclr layer',
                        default=128, type=int)

    parser.add_argument("--local_rank", type=int,
                        default=0, help='Local rank for distributed learning')
    parser.add_argument('--load_path', help='Path to the loading checkpoint',
                        type=str, default="../weight_folder/last.model")
    parser.add_argument("--no_strict", help='Do not strictly load state_dicts',
                        action='store_true')
    parser.add_argument('--batch_size', help='Batch size for test loader',
                        default=2048, type=int)
    parser.add_argument('--save_name', help='save visualized results',
                        type=str, default="visual")
    parser.add_argument('--lambda_p', help='save visualized results',
                        type=float, default=1)
    parser.add_argument('--input_dir', help='input_dir',
                        default="../inputs/", type=str)
    parser.add_argument("-f", type=str, default=1)

    if default:
        return parser.parse_args('')  # empty string
    else:
        return parser.parse_args()

def get_features(loader, P, base_path, train):
    model.eval()
    feats_simclr = []
    feats_shift = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            if train == True :
                x = data['img'][0]  # augmented list of x
            else: x= data['img']
            x = x.to(device)  # gpu tensor
            _, output_aux = model(x, simclr=True, shift=True)
            feats_simclr.append(output_aux['simclr'])
            feats_shift.append(output_aux['shift'])
    feats_simclr = torch.cat(feats_simclr, axis=0)
    feats_shift = torch.cat(feats_shift, axis=0)

    return feats_simclr, feats_shift

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

        
def get_total_scores(scores_id, scores_ood):
    scores = np.concatenate([scores_id, scores_ood])
    labels = np.concatenate([np.ones_like(scores_id), np.zeros_like(scores_ood)])
    return labels, scores
        
def generate_visual(img_dir, ood):
    
    origin_dir = os.path.join('..', P.save_name, label, img_dir.split("/")[-1])
    mask_dir = img_dir+ "_mask.jpg"
    test_img = np.array(Image.open(img_dir).resize((P.cxr_resize,P.cxr_resize)))
    test_mask = np.array(Image.open(mask_dir).resize((P.cxr_resize,P.cxr_resize)))
    test_mask[test_mask>0]=1
    test_masked = np.multiply(test_img, test_mask)
    test_normalized = test_masked
    h_whole = test_normalized.shape[0]  # original w
    w_whole = test_normalized.shape[1]  # original h
    background = np.zeros((h_whole, w_whole))
    background_indicer = np.zeros((h_whole, w_whole))
    sum_prob_wt = 0.0
    model.eval()
    # score_total = []
    for i in range(P.val_iteration):
        non_zero_list = np.nonzero(test_normalized)
        random_index = random.randint(0, len(non_zero_list[0])-1)
        non_zero_row = random.choice(non_zero_list[0]) # random non-zero row index
        non_zero_col = random.choice(non_zero_list[1]) # random non-zero col index
        X_patch = test_img[int(max(0, non_zero_row - (P.cxr_crop / 2))):
                           int(min(h_whole, non_zero_row + (P.cxr_crop / 2))),
                  int(max(0, non_zero_col - (P.cxr_crop / 2))):
                  int(min(w_whole, non_zero_col + (P.cxr_crop / 2)))]
        X_patch_img = data_transforms(augmentation(Image.fromarray(X_patch), rand_p=0.0, mode='test'))
        X_patch_img_ = np.squeeze(np.asarray(X_patch_img))
        X_patch_1 = np.expand_dims(X_patch_img_, axis=0)
        X_patch_2 = np.expand_dims(X_patch_img_, axis=0)
        X_patch_3 = np.expand_dims(X_patch_img_, axis=0)
        X_ = np.concatenate((X_patch_1, X_patch_2, X_patch_3), axis=0)
        X_ = np.expand_dims(X_, axis=0)
        X = torch.from_numpy(X_)
        X = X.to(device)
        _, output_aux = model(X, simclr=True, shift=True)
        score = (output_aux['simclr'][0] * P.axis).sum(dim=1).max().item() * P.weight_simclr
        score += output_aux['shift'][0][0] * P.weight_shiift
        score = score.detach().cpu()

        mask_add = np.zeros((P.cxr_resize, P.cxr_resize))
        mask_add[int(max(0, non_zero_row - (P.cxr_crop / 2))):
                               int(min(h_whole, non_zero_row + (P.cxr_crop / 2))),
            int(max(0, non_zero_col - (P.cxr_crop / 2))):
                      int(min(w_whole, non_zero_col + (P.cxr_crop / 2)))] = score
        indicer = np.ones((P.cxr_crop, P.cxr_crop))
        indicer_size = (int(min(w_whole, non_zero_col + (P.cxr_crop / 2)))
                                          - int(max(0, non_zero_col - (P.cxr_crop / 2))),
                                       int(min(h_whole, non_zero_row + (P.cxr_crop / 2)))
                                       - int(max(0, non_zero_row - (P.cxr_crop / 2)))
                                       )
        indicer = Image.fromarray(indicer).resize(indicer_size, resample=Image.BILINEAR)
        indicer = np.array(indicer)
        indicer_add = np.zeros((P.cxr_resize, P.cxr_resize))
        indicer_add[int(max(0, non_zero_row - (P.cxr_crop / 2))):
                               int(min(h_whole, non_zero_row + (P.cxr_crop / 2))),
            int(max(0, non_zero_col - (P.cxr_crop / 2))):
                      int(min(w_whole, non_zero_col + (P.cxr_crop / 2)))] = indicer
        background = background + mask_add
        background_indicer = background_indicer + indicer_add

    final_mask = np.divide(background, background_indicer + 1e-7)
    cam = final_mask
    cam = cam*test_mask
    nonzero_cam = cam[np.nonzero(cam)].ravel()
    score_q20 = q30 = np.quantile(nonzero_cam,0.20)
    score_min = np.min(cam[np.nonzero(cam)])
    cam_minmax = (cam-score_min)/(cam.max()-score_min)
    cam_minmax = cam_minmax*test_mask
    npy_dir = os.path.join('..', P.save_name, label, "npy",img_dir.split("/")[-1].replace(".png_image.png",".npy"))
    np.save(npy_dir, cam)
    

    red = cm.jet(cam_minmax)[:,:,:3]
    plt_img = test_img/255
    plt_img = np.expand_dims(plt_img, axis=2)
    plt_img = np.concatenate((plt_img, plt_img, plt_img), axis=2)
    plt_img = (red.astype(np.float) + plt_img.astype(np.float))/2
    # print(f'{score:.3f}')

    plt.axis('off')
    plt.grid(b=None)
    plt.imshow(test_img, cmap='gray')
    plt.savefig(origin_dir, bbox_inches = 'tight', pad_inches = 0)
    score_dir = os.path.join('..', P.save_name, label, img_dir.split("/")[-1].replace("image", f"score_{score_q20:.3f}"))
    plt.imshow(plt_img)
    plt.savefig(score_dir, bbox_inches = 'tight', pad_inches = 0)
    plt.clf()
    
P = parse_args()
P.train_type='local'
### Set torch device ###

P.n_gpus = torch.cuda.device_count()
assert P.n_gpus <= 1  # no multi GPU
P.multi_gpu = False

if torch.cuda.is_available():
    torch.cuda.set_device(P.local_rank)
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

for index, P.label in enumerate([0,1,3,2]):
    ## 0: normal // 1: pneumonia // 2: covid-19 // 3: TB
    ### Initialize dataset ###
    test_set, image_size, n_classes = get_dataset(P)

    P.image_size = image_size
    P.n_classes = n_classes

    kwargs = {'pin_memory': False, 'num_workers': P.num_worker}

    test_set = get_subclass_dataset(test_set, classes=P.label)

    ### Initialize model ###
    model = resnet18(num_classes=2).to(device)

    if P.load_path is not None:
        checkpoint = torch.load(P.load_path)
        model.load_state_dict(checkpoint, strict=not P.no_strict)

    data_transforms = transforms.Compose([
                                transforms.Resize((P.cxr_crop, P.cxr_crop)),
                                transforms.ToTensor()])


    base_path ='../weight_folder'  # checkpoint directory

    os.makedirs(os.path.join('..', P.save_name), exist_ok=True)
    png_path = os.path.join('..', P.save_name, str(P.label))
    os.makedirs(png_path, exist_ok=True)
    os.makedirs(os.path.join(png_path, "npy"), exist_ok=True)

    path = base_path + '/train_simclr_features.pth'
    total_train_simclr = torch.load(path)
    path = base_path + '/train_shift_features.pth'
    total_train_shift = torch.load(path)
    P.axis = normalize(total_train_simclr, dim=1).to(device)

    simclr_norm = total_train_simclr.norm(dim=1)
    if P.mode == 'simclr_CSI':
        P.weight_simclr = 1 / simclr_norm.mean().item()
        P.weight_shiift = (1 / total_train_shift[:,0].mean().item()) * P.lambda_p
    else:
        P.weight_simclr = 1
        P.weight_shiift = 0
    print(f'Weight_simclr: {P.weight_simclr: .4f} // Weight_shift: {P.weight_shiift: .4f}')
    print(f'Extraction of label_{P.label} score maps...')

    for idx in tqdm(range(len(test_set))):
        img_dir = test_set[idx]['img_dir']
        label = str(test_set[idx]['label'])
        generate_visual(img_dir, label)
