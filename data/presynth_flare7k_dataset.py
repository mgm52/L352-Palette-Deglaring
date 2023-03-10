import itertools
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
import glob
import random
import time

from scipy import ndimage
import torchvision.transforms.functional as TF
from torch.distributions import Normal
import torch
import numpy as np
import torch
#from basicsr.utils.registry import DATASET_REGISTRY
import matplotlib.pyplot as plt

from data.util.timing import TimeTester

def glod_from_folder(folder_list, index_list):
    ext = ['png','jpeg','jpg','bmp','tif']
    index_dict={}
    for i,folder_name in enumerate(folder_list):
        data_list=[]
        [data_list.extend(glob.glob(folder_name + '/*.' + e)) for e in ext]
        data_list.sort()
        index_dict[index_list[i]]=data_list
    return index_dict

def luminance_mask(img, gamma):
    #calculate mask (the mask is 3 channel)
    one = torch.ones_like(img)
    zero = torch.zeros_like(img)

    luminance=0.3*img[0]+0.59*img[1]+0.11*img[2]
    threshold_value=0.99**gamma
    flare_mask=torch.where(luminance >threshold_value, one, zero)
    return flare_mask

class Flare_Image_Loader_Presynth(data.Dataset):
    def __init__(self, image_path, mask_high_on_lsource=False, mask_type="luminance", randomness=True, gt_is_flare_diff=False):
        gt_dir = os.path.join(image_path, 'gt')
        flare_dir = os.path.join(image_path, 'flare')
        flare_added_dir = os.path.join(image_path, 'flare_added')
        mask_dir = os.path.join(image_path, 'mask')
        self.ext = ['png','jpeg','jpg','bmp','tif']

        self.gt_list=[]
        [self.gt_list.extend(glob.glob(gt_dir + '/*.' + e)) for e in self.ext]
        self.flare_list=[]
        [self.flare_list.extend(glob.glob(flare_dir + '/*.' + e)) for e in self.ext]
        self.data_list=[]
        [self.data_list.extend(glob.glob(flare_added_dir + '/*.' + e)) for e in self.ext]
        self.mask_list=[]
        [self.mask_list.extend(glob.glob(mask_dir + '/*.' + e)) for e in self.ext]

        self.gt_list = sorted(self.gt_list)
        self.flare_list = sorted(self.flare_list)
        self.data_list = sorted(self.data_list)

        self.transform_img = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])

        self.mask_high_on_lsource=mask_high_on_lsource
        self.mask_type=mask_type

        self.randomness = randomness
        self.gt_is_flare_diff = gt_is_flare_diff

        self.val_mode = False

        print("Base images loaded, len:", len(self.data_list))

    def __getitem__(self, index):
        ttimer = TimeTester("Flare_Image_Loader __getitem__", disabled=True) ##
        #t_start = time.process_time()
    	# load base image
        flare_added_path=self.data_list[index]
        flare_path=self.flare_list[index]
        gt_path=self.gt_list[index]
        mask_path=self.mask_list[index]

        to_tensor=transforms.ToTensor()

        flare_added_img = to_tensor(Image.open(flare_added_path).convert('RGB'))
        flare_img = to_tensor(Image.open(flare_path).convert('RGB'))
        gt_img = to_tensor(Image.open(gt_path).convert('RGB'))
        mask_img = to_tensor(Image.open(mask_path).convert('RGB'))
        
        if self.randomness:
            all_img = torch.cat((flare_added_img,flare_img,gt_img,mask_img),0)
            all_img = self.transform_img(all_img)

            flare_added_img = all_img[0:3,:,:]
            flare_img = all_img[3:6,:,:]
            gt_img = all_img[6:9,:,:]
            mask_img = all_img[9:12,:,:]

        if self.gt_is_flare_diff:
            flare_diff = flare_added_img - gt_img
            return_dict = {
                'gt_image': flare_diff,
                'flare': flare_img,
                'cond_image': flare_added_img,
                'path': os.path.basename(flare_added_path),
                'base_image': gt_img,
                'mask': mask_img,
            }
        else:
            return_dict = {
                'gt_image': gt_img,
                'flare': flare_img,
                'cond_image': flare_added_img,
                'path': os.path.basename(flare_added_path),
                'mask': mask_img,
            }

        ttimer.start("computing mask") ##
        #if self.mask_type=="luminance":
        #    flare_mask=luminance_mask(flare_img, self.mask_gamma)
        #    return_dict['mask']=flare_mask
        
        if (return_dict['mask'] is not None) and self.mask_high_on_lsource:
            return_dict['mask'] = 1. - return_dict['mask']
        ttimer.end_prev() ##

        ttimer.end_all() ##
        return return_dict

    def __len__(self):
        return len(self.data_list)
    
    def enable_val_mode(self):
        self.val_mode = True
        self.randomness = False