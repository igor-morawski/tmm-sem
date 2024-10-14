import os
import sys

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image, ImageOps
import glob
import random
import cv2

from scipy import fftpack
from skimage.util import random_noise

random.seed(1143)

import json



# The encoder is not needed in this project and is here because of another project.
# However, you can use this class as a parent to implement your own encoders, 
# e.g. YCbCr encoder, transform-based image encoder etc.

class Encoder():
    def __init__(self, ):
        self.__name__ = "RGB (dummy) Encoder"
               
    def encode(self, img):
        return img

    def decode(self, img):
        return img

    def transpose(self, t):
        t = t.permute(2,0,1)
        return t
    
    def inv_transpose(self, t):
        t = t.permute(1,2,0)
        return t
    
    def extract_dc(self, t):
        return t
    
    def transform_tensor_batch(self, t):
        return t
    
    def inverse_transform_tensor_batch(self, t):
        return t



def populate_train_list(lowlight_images_path, shuffle=True):
    # the default extension in my used datasets is JPG, only png for DARKFACE
    # ideally would be passed as an argument but hardcoded because 
    # of the limited scope of the project
    extension = "JPG"
    if "DARKFACE" in os.path.split(lowlight_images_path)[-1]:
        extension = "png"
    image_list_lowlight = glob.glob(os.path.join(lowlight_images_path, f"*.{extension}"))
    train_list = image_list_lowlight
    if shuffle:
        random.shuffle(train_list)
    return train_list
    

class lowlight_loader(data.Dataset):
    AUGMENTATION_METHODS = ['resize', 'crop', 'crop+resize', 
    'resize+noise', 'crop+noise', 'crop+resize+noise', 
    'nod']

    def __init__(self, lowlight_images_path, encoder, shuffle = True, augmentation_method = 'resize', crop_size = 256, skip_validation=False):
        self.augmentation_method = augmentation_method
        if not self.augmentation_method in self.AUGMENTATION_METHODS:
            raise Exception(f"{self.augmentation_method} not in supported methods ({self.AUGMENTATION_METHODS})")
        self.shuffle = shuffle
        self.train_list = populate_train_list(lowlight_images_path, shuffle=self.shuffle) 
        if not len(self.train_list):
            raise Exception("Dataset empty")
        self.size = crop_size
        
        if not skip_validation:
            print("Validating dataset bitdepth... ") # we use 255 only, change the code to have more options
            for data_lowlight_path in self.train_list:
                if not Image.open(data_lowlight_path).mode == 'RGB':
                    m = Image.open(data_lowlight_path).mode
                    raise Exception(f"{data_lowlight_path} is {m} not RGB ")

        self.data_list = self.train_list
        self.encoder = encoder
        print("Total training examples:", len(self.train_list))
        print(f"Encoder: {self.encoder.__name__}")
        
    def __getitem__(self, index):

        data_lowlight_path = self.data_list[index]
        
        data_lowlight = Image.open(data_lowlight_path)
        
        if self.augmentation_method.split('+noise')[0] == 'resize':
            data_lowlight = data_lowlight.resize((self.size,self.size), Image.ANTIALIAS)
        elif self.augmentation_method.split('+noise')[0] == 'crop':
            if (data_lowlight.width <= self.size) or (data_lowlight.height <= self.size):
                data_lowlight = data_lowlight.resize((self.size,self.size), Image.ANTIALIAS)
            else:
                x = random.randint(0, data_lowlight.width - self.size)
                y = random.randint(0, data_lowlight.height - self.size)
                data_lowlight = data_lowlight.crop((x, y, x + self.size, y + self.size))
        elif self.augmentation_method.split('+noise')[0] == 'crop+resize':
            if (data_lowlight.width <= self.size) or (data_lowlight.height <= self.size):
                data_lowlight = data_lowlight.resize((self.size,self.size), Image.ANTIALIAS)
            else:
                random_crop_size = random.randint(self.size, min(data_lowlight.width, data_lowlight.height))
                x = random.randint(0, data_lowlight.width - random_crop_size)
                y = random.randint(0, data_lowlight.height - random_crop_size)
                data_lowlight = data_lowlight.crop((x, y, x + random_crop_size, y + random_crop_size))
                data_lowlight = data_lowlight.resize((self.size,self.size), Image.ANTIALIAS)
        elif self.augmentation_method == 'nod':
                original_width, original_height = data_lowlight.size
                max_dim = max(original_width, original_height)
                scale = self.size / max_dim
                data_lowlight = data_lowlight.resize((int(scale * data_lowlight.width),
                                      int(scale * data_lowlight.height)), Image.ANTIALIAS)
                width, height = data_lowlight.size
                w_pad, h_pad = self.size - width, self.size - height
                left_pad, right_pad, top_pad, bottom_pad = w_pad//2, w_pad - w_pad//2, h_pad // 2, h_pad - h_pad // 2
                a = np.array(data_lowlight)
                a = np.pad(a, ((top_pad, bottom_pad), (right_pad, left_pad), (0, 0)), mode='reflect')
                data_lowlight = Image.fromarray(a)
        else:
            raise NotImplementedError

        data_lowlight = (np.asarray(data_lowlight)/255.0)     
        if '+noise' in self.augmentation_method:   
            data_lowlight = np.clip(random_noise(data_lowlight, var=0.15**2), 0, 1)

        data_lowlight = self.encoder.encode(data_lowlight) 
        data_lowlight = torch.from_numpy(data_lowlight).float()
        data_lowlight = self.encoder.transpose(data_lowlight)
        return data_lowlight

    def __len__(self):
        return len(self.data_list)


def pad_to_divisble_by(img, size):
    h, w, c = img.shape
    pad_h = (size - h % size) % size
    pad_w = (size - w % size) % size
    if pad_h == 0 and pad_w == 0:
        return img

    # Pad the image with mirrored pixels on the right and bottom edges
    pad_img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)

    return pad_img

ENCODER_NAME_TO_ENCODER = {"RGB" : Encoder, }

