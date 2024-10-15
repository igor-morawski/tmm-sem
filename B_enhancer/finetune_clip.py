import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim

import os
import sys
import argparse
import time
import numpy as np
import json

from datetime import datetime
from tensorboardX import SummaryWriter

import myloss
import dataloader
from settings import *
from CLIP import clip
from transformers import CLIPProcessor, CLIPModel


import random
random.seed(11)

PARS_SEPARATOR = ','
PARS_SEPARATOR_2 = '+'
assert PARS_SEPARATOR != PARS_SEPARATOR_2


def rename_duplicates(loss_names):
    name_count = {}
    renamed_names = []
    
    for name in loss_names:
        if name in name_count:
            name_count[name] += 1
            new_name = f"{name}{name_count[name]}"
        else:
            name_count[name] = 0
            new_name = name
        renamed_names.append(new_name)

    return renamed_names
    

def train(config):
    
    now = datetime.now()
    date_string = now.strftime("%y%m%d_%H%M%S")
    dir_name = date_string
    dir_name += f"_{os.path.split(config.lowlight_images_path)[-1]}"
    dir_name += f"_batch{config.train_batch_size}x{config.crop_size}"
    if config.name:
        dir_name += f"_{config.name}"
    experiment_dir = os.path.join(config.snapshots_folder, dir_name)
    if not os.path.exists(experiment_dir):
       os.mkdir(experiment_dir)
    with open(os.path.join(experiment_dir, "args.json"), 'w') as f: 
       json.dump(vars(config), f, indent=4)
    writer = SummaryWriter(os.path.join(experiment_dir, "logs"))
    
    if '+' in config.lowlight_images_path:
        config.lowlight_images_path = config.lowlight_images_path.split("+")
    else:
        config.lowlight_images_path = [config.lowlight_images_path, ]
        assert len(config.lowlight_images_path) == 1
    assert isinstance(config.lowlight_images_path, list)
    loader = dataloader.DATALOADER_NAME_TO_DATALOADER[config.dataloader_name]
    train_dataset = loader(lowlight_images_path = config.lowlight_images_path,
                           datasets_roots = config.lowlight_images_path, # will be dispatched depending on the dataloader used
                           json_bboxes_fp = config.bbox_json,
                                               augmentation_method = config.augmentation,
                                               crop_size = config.crop_size,
                                               skip_validation=True
                                               )		
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config.train_batch_size,
                                               shuffle=True,
                                               num_workers=config.num_workers,
                                               pin_memory=True)
            
    loss_names = config.loss_names.split(PARS_SEPARATOR) 
    loss_weights = [float(_w) for _w in config.loss_weights.split(PARS_SEPARATOR)]
    loss_inputs = [_chunk.split(PARS_SEPARATOR_2) for _chunk in config.loss_inputs.split(PARS_SEPARATOR)]
    assert len(loss_names) == len(loss_weights) == len(loss_inputs)
    
    trainable_clip_inference = None
    if any([l in loss_names for l in ('L_trainable_CLIP_img_txt', 'L_trainable_CLIP_img')]):
        trainable_clip_model = CLIPModel.from_pretrained("./clip-vit-base-patch32")
        trainable_clip_processor = CLIPProcessor.from_pretrained("./clip-vit-base-patch32") 
        trainable_clip_model = trainable_clip_model.cuda()
        trainable_clip_model.eval()
        
        for n, p in trainable_clip_model.named_parameters():
            if ('visual_projection' in n) or ('text_projection' in n):
                p.requires_grad = True
            else:
                p.requires_grad = False

        clip_mean = torch.Tensor([0.48145466, 0.4578275, 0.40821073]).to("cuda").unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        clip_std = torch.Tensor([0.26862954, 0.26130258, 0.27577711]).to("cuda").unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        f_images2clip = lambda x: (torch.nn.functional.interpolate(x, [224, 224], mode = 'bilinear') - clip_mean) / clip_std
        
        def trainable_clip_inference(imgs, txts):
            inputs = trainable_clip_processor(text=txts, return_tensors="pt", padding=True, truncation=True, max_length=77)
            inputs["input_ids"] = inputs["input_ids"].to("cuda")
            inputs["attention_mask"] = inputs["attention_mask"].to("cuda")
            inputs["pixel_values"] = f_images2clip(imgs)
            result = trainable_clip_model(**inputs)
            return result
      
    loss_optional_kwargs = {
                            'trainable_clip_inference' : trainable_clip_inference
                            }
    
    loss_objs = [myloss.LOSS_NAME_TO_LOSS[_loss_name](weight = _loss_weight, **loss_optional_kwargs) \
        for _loss_name, _loss_weight in \
            zip(loss_names,
                loss_weights)]
    
    optimizer = torch.optim.Adam(trainable_clip_model.parameters(), lr=config.lr, )
    
    loss_names = rename_duplicates(loss_names)
    
    idx = 0
    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch}, iterations left: {len(train_loader)} ")
        for iteration, batch_data in enumerate(train_loader):
            
            total_loss = 0
            
            # batch_data['low']  = batch_data['imgs'][0]
            batch_data['low']  = random.choice(batch_data['imgs'])
            batch_data['low'] = batch_data['low'].cuda()
            
            for loss_obj, loss_name, loss_input in \
                zip(loss_objs, loss_names, loss_inputs):
                loss_val = loss_obj( *[batch_data[_key] for _key in loss_input] )
                writer.add_scalar(f'loss/loss_{loss_name}', loss_val, idx)
                total_loss += loss_val
                
            writer.add_scalar('loss/loss', total_loss , idx)

            idx += 1
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            state_dict_filtered = {}
            for n in trainable_clip_model.state_dict().keys():
                if ('visual_projection' in n) or ('text_projection' in n):
                    state_dict_filtered[n] = trainable_clip_model.state_dict()[n]
                    
            if ((idx) % config.display_iter) == 0:
                print("Loss at iteration", iteration+1, ":", f"({idx})", total_loss.item())
            if ((idx) % config.snapshot_iter) == 0:
                torch.save(state_dict_filtered, os.path.join(experiment_dir, "Epoch" + str(epoch) + '.pth')) 	
                print("Saving at iteration", iteration+1, ":", f"({idx})", total_loss.item())
        


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--lowlight_images_path', type=str, default="data/train_data/")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--snapshots_folder', type=str, default="stage0_train_clip_results/")
    parser.add_argument('--augmentation', required=True)
    parser.add_argument('--name', type=str)
    parser.add_argument('--crop_size', type=int, default=256)

    parser.add_argument('--dataloader_name', type=str, required=True)
    parser.add_argument('--bbox_json', type=str, default=None)
    
    parser.add_argument('--loss_names', type=str, help='Separated by {PARS_SEPARATOR}', required=True)
    parser.add_argument('--loss_weights', type=str, help='Separated by {PARS_SEPARATOR}', required=True)
    parser.add_argument('--loss_inputs', type=str, help='Separated by {PARS_SEPARATOR} for each loss and {PARS_SEPARATOR2}', required=True)

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)

    train(config)

