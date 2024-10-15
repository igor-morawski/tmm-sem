import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import argparse
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import json
import tqdm

import cv2
 
from model import ModelProcessor
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('cpt', type=str)
    parser.add_argument('--pattern', default="*", type=str)
    parser.add_argument('--intermediate', action="store_true")
    parser.add_argument('--data_dir', default='test_data/')
    parser.add_argument('--scale', default=1., type=float)
    parser.add_argument('--out_dir', default='out', )
    
    config = parser.parse_args()
    
    resize_wrapper = lambda x: x
    if config.scale != 1.:
        resize_wrapper = lambda x: cv2.resize(x, (int(x.shape[1] * config.scale), int(x.shape[0] * config.scale)))
    print(f"Testing {config.cpt}")
    
    with open(os.path.join(os.path.split(config.cpt)[0], "args.json"), "r") as f:
        args_data = json.load(f)
        
    for key, value in args_data.items():
        if key in config:
            continue
        setattr(config, key, value)
    
    model_args = {}
    net = model.MODEL_NAME_TO_MODEL[config.model](**model_args).cuda()
    net.load_state_dict(torch.load(config.cpt))
    net.eval()

    model_processor = ModelProcessor(net, additional_preprocessing_pipeline = [resize_wrapper])

    model_dir = os.path.split(config.cpt)[0]    
    if not config.out_dir:
        out_dir = os.path.join(model_dir, f"results_{os.path.split(config.cpt)[-1].split('.')[0]}")
    else:
        out_dir = config.out_dir
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        assert os.path.exists(out_dir)
        
    def test(image_path):
        enhanced = model_processor.infere_filepath(image_path)
        filename = os.path.split(image_path)[-1]
        cv2.imwrite(os.path.join(out_dir, filename), enhanced[0].permute(1, 2, 0).detach().cpu().numpy()[...,::-1] * 255)
        
    
    print(f"Infering {config.data_dir}")
    for image_path in tqdm.tqdm(glob.glob(os.path.join(config.data_dir, config.pattern))):
        with torch.no_grad():
            test(image_path, )

