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
from skimage import morphology
from skimage.measure import label
from skimage.filters import rank
from skimage.morphology import disk
from skimage import color
from skimage.measure import regionprops
import torchvision.transforms.functional as TF
from torch.distributions import Normal
import torch
import numpy as np
import torch
#from basicsr.utils.registry import DATASET_REGISTRY
import matplotlib.pyplot as plt
import cv2

from data.util.timing import TimeTester

def plot_light_pos(input_img,threshold):
    #input should be a three channel tensor with shape [C,H,W]
    #Out put the position (x,y) in int
    luminance=0.3*input_img[0]+0.59*input_img[1]+0.11*input_img[2]
    luminance_mask=luminance>threshold
    luminance_mask_np=luminance_mask.numpy()
    struc = disk(3)
    img_e = ndimage.binary_erosion(luminance_mask_np, structure = struc)
    img_ed = ndimage.binary_dilation(img_e, structure = struc)

    labels = label(img_ed)
    if labels.max() == 0:
        print("Light source not found.")
        return None
    else:
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        largestCC=largestCC.astype(int)
        properties = regionprops(largestCC, largestCC)
        weighted_center_of_mass = properties[0].weighted_centroid
        x, y = int(weighted_center_of_mass[1]), int(weighted_center_of_mass[0])
        print("Light source detected in position: x:",x,",y:",y)
        return (x,y)


class RandomGammaCorrection(object):
    def __init__(self, gamma = None):
        self.gamma = gamma
    def __call__(self,image):

        if self.gamma == None:
            # more chances of selecting 0 (original image)
            gammas = [0.5,1,2]
            self.gamma = random.choice(gammas)
            # apply to each image in image
            results = [TF.adjust_gamma(image[i:i+3], self.gamma, gain=1) for i in range(0, image.shape[0], 3)]
            return torch.cat(results, dim=0)
        elif isinstance(self.gamma,tuple):
            gamma=random.uniform(*self.gamma)
            results = [TF.adjust_gamma(image[i:i+3], gamma, gain=1) for i in range(0, image.shape[0], 3)]
            return torch.cat(results, dim=0)
        elif self.gamma == 0:
            return image
        else:
            results = [TF.adjust_gamma(image[i:i+3],self.gamma,gain=1) for i in range(0, image.shape[0], 3)] 
            return torch.cat(results, dim=0)

class TranslationTransform(object):
    def __init__(self, position):
        self.position = position

    def __call__(self, x):
        return TF.affine(x,angle=0, scale=1,shear=[0,0], translate= list(self.position))

def remove_background(image):
    #the input of the image is PIL.Image form with [H,W,C]
    image=np.float32(np.array(image))
    _EPS=1e-7
    rgb_max=np.max(image,(0,1))
    rgb_min=np.min(image,(0,1))
    image=(image-rgb_min)*rgb_max/(rgb_max-rgb_min+_EPS)
    image=torch.from_numpy(image)
    return image

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


# Transform that has an x% chance of setting the dominant hue in an image to a particular value
class ColourBias(torch.nn.Module):
    def __init__(self, biases, chance=1):
        super().__init__()
        self.biases = biases
        self.chance = chance

    def forward(self, img):
        if random.random() < self.chance:
            img = img.permute(1, 2, 0)
            img = img.cpu().numpy()
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            [bias_low, bias_high] = random.choice(self.biases)
            bias = random.uniform(bias_low, bias_high)
            #print(f"Setting hue bias to {bias}")
            hsv[:, :, 0] = bias * 0.99 + hsv[:, :, 0] * 0.01
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            img = torch.from_numpy(img).permute(2, 0, 1)
        return img

class Flare_Image_Loader(data.Dataset):
    def __init__(self, image_path ,transform_base,transform_flare,mask_type=None,mask_high_on_lsource=True,placement_mode="random"):
        self.ext = ['png','jpeg','jpg','bmp','tif']
        self.data_list=[]
        [self.data_list.extend(glob.glob(image_path + '/*.' + e)) for e in self.ext]
        self.flare_dict={}
        self.flare_list=[]
        self.flare_name_list=[]
        self.lsource_list=[]
        self.using_lsources=False
        self.flare_and_lsource_list=[]

        self.placement_mode=placement_mode

        self.reflective_flag=False
        self.reflective_dict={}
        self.reflective_list=[]
        self.reflective_name_list=[]

        self.colour_bias = ColourBias([[15, 60], [200, 245]], 0.66)

        self.mask_type=mask_type #It is a str which may be None,"luminance" or "color"
        self.mask_high_on_lsource=mask_high_on_lsource
        self.img_size=transform_base['img_size']
        self.transform_base=transforms.Compose(
            ([transforms.Resize(transform_base['pre_crop_size'])] if (transform_base is not None) else []) + [
            transforms.RandomCrop((self.img_size,self.img_size),pad_if_needed=True,padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
            ])

        self.transform_flare_affine=transforms.RandomAffine(
                degrees=(0,360),
                scale=(transform_flare['scale_min'],transform_flare['scale_max']),                             # default: (0.8, 1.5)
                translate=(0,0),#(transform_flare['translate']/1440,transform_flare['translate']/1440),    
                shear=(-transform_flare['shear'],transform_flare['shear']))                                    # default: (-20, 20)                    

        print("Base images loaded, len:", len(self.data_list))

    def __getitem__(self, index):
        ttimer = TimeTester("Flare_Image_Loader __getitem__", disabled=True) ##
        #t_start = time.process_time()
    	# load base image
        img_path=self.data_list[index]

        ttimer.start("loading base image") ##
        base_img = Image.open(img_path).convert('RGB')
        ttimer.end_prev() ##
        
        ttimer.start("transforming base image") ##
        gamma=np.random.uniform(1.8,2.2)
        to_tensor=transforms.ToTensor()
        adjust_gamma=RandomGammaCorrection(gamma)
        adjust_gamma_reverse=RandomGammaCorrection(1/gamma)
        color_jitter=transforms.ColorJitter(brightness=(0.8,3),hue=0.0, saturation=(1, 1))#(0.25, 0.8))
        if self.transform_base is not None:
            base_img=to_tensor(base_img)
            base_img=adjust_gamma(base_img)
            base_img=self.transform_base(base_img)
        else:
            base_img=to_tensor(base_img)
            base_img=adjust_gamma(base_img)
            base_img=base_img.permute(2,0,1)
        sigma_chi=0.01*np.random.chisquare(df=1)
        base_img=Normal(base_img,sigma_chi).sample()
        gain=np.random.uniform(1,1.2)
        flare_DC_offset=np.random.uniform(-0.02,0.02)
        base_img=gain*base_img
        base_img=torch.clamp(base_img,min=0,max=1)
        ttimer.end_prev() ##

        #t_lpos_start = time.process_time()
        if self.placement_mode == "centre":
            light_pos=[0,0]
        else:
            if self.placement_mode == "light_pos":
                light_pos=plot_light_pos(base_img,0.97**gamma)
            if (self.placement_mode == "random") or (light_pos is None):
                # Set random light pos
                light_pos=[np.random.randint(0,base_img.shape[1]),np.random.randint(0,base_img.shape[1])]
            light_pos=[light_pos[0]-base_img.shape[1]/2,light_pos[1]-base_img.shape[1]/2]
        #t_lpos_stop = time.process_time() - t_lpos_start

        #load flare image
        if self.using_lsources:
            (flare_path, lsource_path)=random.choice(self.flare_and_lsource_list)
            ttimer.start("loading lsource and flare image") ##
            lsource_img=to_tensor(Image.open(lsource_path).convert('RGB'))
            flare_img=to_tensor(Image.open(flare_path).convert('RGB'))
            ttimer.end_prev() ##
            flare_and_lsource = torch.cat((flare_img,lsource_img),0)
        else:
            flare_path=random.choice(self.flare_list)
            ttimer.start("loading flare image") ##
            flare_img=to_tensor(Image.open(flare_path).convert('RGB'))
            ttimer.end_prev() ##
            flare_and_lsource = flare_img
        if self.reflective_flag:
            reflective_path=random.choice(self.reflective_list)
            reflective_img =Image.open(reflective_path).convert('RGB')

        ttimer.start("flare and lsource gamma correction") ##
        # this is really costly!
        flare_and_lsource=adjust_gamma(flare_and_lsource)
        ttimer.end_prev() ##
        
        if self.reflective_flag:
            reflective_img=to_tensor(reflective_img)
            reflective_img=adjust_gamma(reflective_img)
            flare_and_lsource[0:3,:,:] = torch.clamp(flare_and_lsource[0:3,:,:]+reflective_img,min=0,max=1)

        ttimer.start("removing background") ##
        flare_and_lsource[0:3,:,:]=remove_background(flare_and_lsource[0:3,:,:])
        ttimer.end_prev() ##

        ttimer.start("transforming flare") ##
        # this whole transform is really costly!
        transform_flare=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            self.transform_flare_affine, # <-- this is really costly!
            TranslationTransform(light_pos),
            transforms.CenterCrop((self.img_size,self.img_size)),
        ])
        flare_and_lsource = transform_flare(flare_and_lsource)
        ttimer.end_prev() ##

        ttimer.start("dividing flare and lsource") ##
        if self.using_lsources:
            flare_img = flare_and_lsource[0:3,:,:]
            lsource_img = flare_and_lsource[3:6,:,:]
        ttimer.end_prev() ##

        #change color
        ttimer.start("changing color") ##
        flare_img=color_jitter(flare_img)
        flare_img=self.colour_bias(flare_img)
        ttimer.end_prev() ##

        #flare blur
        ttimer.start("blurring flare") ##
        blur_transform=transforms.GaussianBlur(21,sigma=(0.1,3.0))
        flare_img=blur_transform(flare_img)
        flare_img=flare_img+flare_DC_offset
        flare_img=torch.clamp(flare_img,min=0,max=1)
        ttimer.end_prev() ##

        #merge image    
        merge_img=flare_img+base_img
        merge_img=torch.clamp(merge_img,min=0,max=1)

        ttimer.start("reversing gamma") ##
        return_dict = {
			'gt_image': adjust_gamma_reverse(base_img),
			'flare': adjust_gamma_reverse(flare_img),
			'cond_image': adjust_gamma_reverse(merge_img),
			'gamma':gamma,
			'path': os.path.basename(img_path),
		}

        if self.using_lsources:
            return_dict['lsource'] = adjust_gamma_reverse(lsource_img)
        ttimer.end_prev() ##

        ttimer.start("computing mask") ##
        if self.mask_type=="luminance":
            flare_mask=luminance_mask(flare_img, gamma)

            if self.using_lsources:
                lsource_mask=luminance_mask(lsource_img, gamma)
                return_dict['lsource_mask']=lsource_mask
                return_dict['flare_mask']=flare_mask
                return_dict['mask']=torch.clamp(flare_mask+lsource_mask, min=0, max=1)
            else:
                return_dict['mask']=flare_mask
        elif self.mask_type=="color":
            one = torch.ones_like(base_img)
            zero = torch.zeros_like(base_img)

            threshold_value=0.99**gamma
            flare_mask=torch.where(merge_img >threshold_value, one, zero)
            return_dict['mask']=flare_mask
        
        if (return_dict['mask'] is not None) and not self.mask_high_on_lsource:
            return_dict['mask'] = 1. - return_dict['mask']
        ttimer.end_prev() ##

        ttimer.end_all() ##
        return return_dict

    def __len__(self):
        return len(self.data_list)
    
    def load_scattering_flare(self,flare_name,flare_path,lsource_path=None):
        flare_list=[]
        [flare_list.extend(glob.glob(flare_path + '/*.' + e)) for e in self.ext]
        self.flare_name_list.append(flare_name)
        self.flare_dict[flare_name]=flare_list
        self.flare_list.extend(flare_list)
        len_flare_list=len(self.flare_dict[flare_name])

        if lsource_path is not None:
            self.using_lsources = True
            lsource_list=[]
            [lsource_list.extend(glob.glob(lsource_path + '/*.' + e)) for e in self.ext]
            self.lsource_list.extend(lsource_list)
            len_lsource_list=len(lsource_list)

            self.flare_and_lsource_list.extend(list(itertools.product(flare_list,lsource_list)))
        else:
            self.using_lsources = False

        if len_flare_list == 0 or (self.using_lsources and len_lsource_list == 0):
            print("ERROR: scattering flare images are not loaded properly")
        else:
            print("Scattering Flare Image:",flare_name, " is loaded successfully with examples", str(len_flare_list))
        print("Now we have",len(self.flare_list),'scattering flare images, with',len(self.lsource_list),'light sources')

    def load_reflective_flare(self,reflective_name,reflective_path):
        self.reflective_flag=True
        reflective_list=[]
        [reflective_list.extend(glob.glob(reflective_path + '/*.' + e)) for e in self.ext]
        self.reflective_name_list.append(reflective_name)
        self.reflective_dict[reflective_name]=reflective_list
        self.reflective_list.extend(reflective_list)
        len_reflective_list=len(self.reflective_dict[reflective_name])
        if len_reflective_list == 0:
            print("ERROR: reflective flare images are not loaded properly")
        else:
            print("Reflective Flare Image:",reflective_name, " is loaded successfully with examples", str(len_reflective_list))
        print("Now we have",len(self.reflective_list),'refelctive flare images')

#@DATASET_REGISTRY.register()
class Flare_Pair_Loader(Flare_Image_Loader):
    def __init__(self, opt):
        Flare_Image_Loader.__init__(self,opt['bg_path'],opt['transform_base'],opt['transform_flare'],opt['mask_type'],opt['mask_high_on_lsource'],opt['placement_mode'])
        self.load_scattering_flare("all_flares",opt['flare_path'],opt['lsource_path'])
        #reflective_dict=opt['reflective_dict']
        #if len(scattering_dict) !=0:
        #    for key in scattering_dict.keys():
                
        #if len(reflective_dict) !=0:
        #    for key in reflective_dict.keys():
        #        self.load_reflective_flare(key,reflective_dict[key])

#@DATASET_REGISTRY.register()
class Image_Pair_Loader(data.Dataset):
    def __init__(self, opt):
        super(Image_Pair_Loader, self).__init__()
        self.opt = opt
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        self.paths = glod_from_folder([self.lq_folder, self.gt_folder], ['cond_image', 'gt_image'])
        self.to_tensor = transforms.ToTensor()
        self.gt_size = opt['gt_size']
        self.transform = transforms.Compose([transforms.Resize(self.gt_size), transforms.CenterCrop(self.gt_size), transforms.ToTensor()])

    def __getitem__(self, index):
        gt_path = self.paths['gt_image'][index]
        cond_image_path = self.paths['cond_image'][index]
        img_cond_image=self.transform(Image.open(cond_image_path).convert('RGB'))
        img_gt=self.transform(Image.open(gt_path).convert('RGB'))

        return {'cond_image': img_cond_image, 'gt_image': img_gt}

    def __len__(self):
        return len(self.paths)