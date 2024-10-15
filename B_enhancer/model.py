from settings import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#import pytorch_colors as colors
import numpy as np

from PIL import Image
import cv2 

class ModelProcessor(nn.Module):
    def __init__(self, net, padding_size = 32, additional_preprocessing_pipeline = []):
        super(ModelProcessor, self).__init__()
        self.net = net 
        self.padding_size = padding_size 
        self.additional_preprocessing_pipeline = additional_preprocessing_pipeline
        
    def infere_filepath(self, filepath):
        if 'TIFF' in filepath:
            array = cv2.imread(filepath, cv2.IMREAD_UNCHANGED) * 255.0
            return self.infere_array_RGB255(array)
        pil_img = Image.open(filepath)
        if (pil_img.mode == 'L') or (pil_img.mode == 'P') or ('A' in pil_img.mode):
            pil_img = pil_img.convert('RGB')
        array = np.asarray(pil_img)
        return self.infere_array_RGB255(array)
    
    def infere_array_RGB255(self, array):
        array = (array/255.0) 
        return self.infere_array_RGB01(array)
    
    def pad_array_to_divisble_by(self, img, size):
        h, w, c = img.shape
        pad_h = (size - h % size) % size
        pad_w = (size - w % size) % size
        if pad_h == 0 and pad_w == 0:
            return img
        pad_img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        return pad_img
    
    def tensor_pad_to_divisble_by(self, img, size):
        b, c, h, w = img.size()
        pad_h = (size - h % size) % size
        pad_w = (size - w % size) % size
        if pad_h == 0 and pad_w == 0:
            return img

        # Pad the image with mirrored pixels on the right and bottom edges
        pad_img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode = 'reflect')

        return pad_img

    def apply_preprocessing_pipeline(self, array):
        if self.additional_preprocessing_pipeline: 
            for f in self.additional_preprocessing_pipeline:
                array = f(array)
        return array 
    
    def cvt_array2tensor_RGB01(self, array):
        img = torch.from_numpy(array).float()
        img = img.permute(2, 0, 1)
        img = img.unsqueeze(0)
        return img 
    
    def infere_array_RGB01(self, array):
        array = self.apply_preprocessing_pipeline(array)
        img = self.cvt_array2tensor_RGB01(array)
        b, c, h, w = img.size() 
        img = self.tensor_pad_to_divisble_by(img, 32)
        result = self.net(img.cuda())
        result = result[:, :, :h, :w]
        result = torch.clamp(result, 0, 1) 
        return result

class IdentityModel(nn.Module):

    def __init__(self, **kwargs):
        super(IdentityModel, self).__init__()
    
    def forward(self, x, **kwargs):
        return x
    

class ZDCE(nn.Module):

    def __init__(self, number_f=32, input_ch=3, output_ch=24, verbose=True, ):
        super(ZDCE, self).__init__()

        self.verbose = True

        self.relu = nn.ReLU(inplace=True)

        self.input_ch = input_ch
        self.output_ch = output_ch

        self.number_f = number_f

        number_f = number_f
        self.e_conv1 = nn.Conv2d(self.input_ch, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f*2, number_f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(number_f*2, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(
            number_f*2, self.output_ch, 3, 1, 1, bias=True)

        self.maxpool = nn.MaxPool2d(
            2, stride=2, return_indices=False, ceil_mode=False)
        
        self.apply(self.weights_init)
        
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def full_forward(self, x, ):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))

        x_r = F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)

        x = x + r1*(torch.pow(x, 2)-x)
        x = x + r2*(torch.pow(x, 2)-x)
        x = x + r3*(torch.pow(x, 2)-x)
        enhance_image_1 = x + r4*(torch.pow(x, 2)-x)
        x = enhance_image_1 + r5 * \
            (torch.pow(enhance_image_1, 2)-enhance_image_1)
        x = x + r6*(torch.pow(x, 2)-x)
        x = x + r7*(torch.pow(x, 2)-x)
        enhance_image = x + r8*(torch.pow(x, 2)-x)
        r = torch.cat([r1, r2, r3, r4, r5, r6, r7, r8], 1)
        
        return {'enhanced_image_1_full': enhance_image_1, 
                NET_ENHANCED_KEY : enhance_image, 
                'A' : r}
    
    def forward(self, x, ):
        return self.full_forward(x)[NET_ENHANCED_KEY]

    
MODEL_NAME_TO_MODEL = {'IdentityModel':IdentityModel,
                       "ZDCE": ZDCE,}


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from add_models import added_models_dict
    for k, v in added_models_dict.items():
        MODEL_NAME_TO_MODEL[k] = v
    print("Loaded extra models")
except ModuleNotFoundError:
    print("You can add extra modelss in add_models")
    ''' You can add your extra following this file structure, 
    first implementing your model and then adding
    its class in added_models_dict that follows this module's
    MODEL_NAME_TO_MODEL '''
warnings.resetwarnings()

