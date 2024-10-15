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

import model
import myloss
import dataloader
from settings import *
from CLIP import clip
from transformers import CLIPProcessor, CLIPModel

from operator import attrgetter

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
    if config.checkpoint:
        checkpoint_name = os.path.split(os.path.split(config.checkpoint)[0])[-1]
        dir_name += f"_from_{checkpoint_name}_to"
    dir_name += f"_CLIP_{config.model}"
    dir_name += f"_{os.path.split(config.lowlight_images_path)[-1]}"
    dir_name += f"_batch{config.train_batch_size}x{config.crop_size}"
    if config.name:
        dir_name += f"_{config.name}"
        
    if config.resume:
        dir_name = os.path.split(os.path.split(config.resume)[0])[-1]
        
    experiment_dir = os.path.join(config.snapshots_folder, dir_name)
    if config.resume:
        assert os.path.exists(config.resume)
        assert os.path.exists(experiment_dir)
        
        with open(os.path.join(experiment_dir, "args.json"), 'r') as f: 
            _config = json.load(f)
            _d_config_new = vars(config)
            for _key in _config.keys():
                if _key == 'resume' : continue
                assert _d_config_new[_key] == _config[_key]
    
    if not os.path.exists(experiment_dir):
       os.mkdir(experiment_dir)
    with open(os.path.join(experiment_dir, "args.json"), 'w') as f: 
       json.dump(vars(config), f, indent=4)
    writer = SummaryWriter(os.path.join(experiment_dir, "logs"))

    model_args = {}
    if "ZDCE" in config.model: model_args["number_f"] = getattr(config, "number_f", 32)
    
    net = model.MODEL_NAME_TO_MODEL[config.model](**model_args).cuda()

    loader = dataloader.DATALOADER_NAME_TO_DATALOADER[config.dataloader_name]
    dataset_roots = [config.lowlight_images_path, ] if "," not in config.lowlight_images_path else config.lowlight_images_path.split(",")
    train_dataset = loader(lowlight_images_path = config.lowlight_images_path,
                           datasets_roots = dataset_roots, # will be dispatched depending on the dataloader used
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

    if config.checkpoint:
        net.load_state_dict(torch.load(config.checkpoint), strict=True)
        print("Checkpoint loaded")

    if config.resume:
        net.load_state_dict(torch.load(config.resume))
        print(f"Resuming from {config.resume}")
    
    net.train()
            
    loss_names = config.loss_names.split(PARS_SEPARATOR) 
    loss_weights = [float(_w) for _w in config.loss_weights.split(PARS_SEPARATOR)]
    loss_inputs = [_chunk.split(PARS_SEPARATOR_2) for _chunk in config.loss_inputs.split(PARS_SEPARATOR)]
    assert len(loss_names) == len(loss_weights) == len(loss_inputs)
    
    
    label_clip_model, label_processor = None, None
    clip_inference = None 
    if any([l in loss_names for l in ('L_CLIP_BCE_Label', )]):
        label_clip_model = CLIPModel.from_pretrained("./clip-vit-base-patch32")
        label_processor = CLIPProcessor.from_pretrained("./clip-vit-base-patch32") 
        label_clip_model = label_clip_model.to("cuda")   
    
    prompt_clip_model, prompt_processor = None, None
    if any([l in loss_names for l in ('L_CLIP_Pretrained_Prompt', )]):
        prompt_clip_model, prompt_processor = clip.load("ViT-B/32", device=torch.device("cuda"), download_root="./clip_model/") 
        prompt_clip_model = prompt_clip_model.to(torch.float32)
    
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
    
    if any([l in loss_names for l in ('LCLIP_img_txt', 'LCLIP_img', )]):
        clip_model = CLIPModel.from_pretrained("./clip-vit-base-patch32")
        clip_processor = CLIPProcessor.from_pretrained("./clip-vit-base-patch32") 
        clip_model = clip_model.cuda()
        clip_model.eval()
        
        for p in clip_model.parameters():
            p.requires_grad = False

        clip_mean = torch.Tensor([0.48145466, 0.4578275, 0.40821073]).to("cuda").unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        clip_std = torch.Tensor([0.26862954, 0.26130258, 0.27577711]).to("cuda").unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        f_images2clip = lambda x: (torch.nn.functional.interpolate(x, [224, 224], mode = 'bilinear') - clip_mean) / clip_std
        
        def clip_inference(imgs, txts):
            inputs = clip_processor(text=txts, return_tensors="pt", padding=True, truncation=True, max_length=77)
            inputs["input_ids"] = inputs["input_ids"].to("cuda")
            inputs["attention_mask"] = inputs["attention_mask"].to("cuda")
            inputs["pixel_values"] = f_images2clip(imgs)
            result = clip_model(**inputs)
            return result
        
    condition_content_finetuned = any([l in loss_names for l in ('LCLIP_img_txt_finetuned_content', 'LCLIP_img_finetuned_content', )])
    clip_content_ftinference = None 
    if config.checkpoint_content_clip_projection or condition_content_finetuned :
        assert config.checkpoint_content_clip_projection
        assert condition_content_finetuned
        clip_content_ftmodel = CLIPModel.from_pretrained("./clip-vit-base-patch32")
        clip_content_ftprocessor = CLIPProcessor.from_pretrained("./clip-vit-base-patch32") 
        clip_content_ftmodel = clip_content_ftmodel.cuda()
        # LOAD
        clip_content_ftmodel.load_state_dict(torch.load(config.checkpoint_content_clip_projection), strict=False)
        # END LOAD
        clip_content_ftmodel.eval()
        
        for p in clip_content_ftmodel.parameters():
            p.requires_grad = False

        clip_content_ftmean = torch.Tensor([0.48145466, 0.4578275, 0.40821073]).to("cuda").unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        clip_content_ftstd = torch.Tensor([0.26862954, 0.26130258, 0.27577711]).to("cuda").unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        f_images2clip = lambda x: (torch.nn.functional.interpolate(x, [224, 224], mode = 'bilinear') - clip_content_ftmean) / clip_content_ftstd
        
        def clip_content_ftinference(imgs, txts):
            inputs = clip_content_ftprocessor(text=txts, return_tensors="pt", padding=True, truncation=True, max_length=77)
            inputs["input_ids"] = inputs["input_ids"].to("cuda")
            inputs["attention_mask"] = inputs["attention_mask"].to("cuda")
            inputs["pixel_values"] = f_images2clip(imgs)
            result = clip_content_ftmodel(**inputs)
            return result
        
    condition_context_finetuned = any([l in loss_names for l in ('LCLIP_img_txt_finetuned_context', 'LCLIP_img_finetuned_context', )])
    clip_context_ftinference = None 
    if config.checkpoint_context_clip_projection or condition_context_finetuned :
        assert config.checkpoint_context_clip_projection
        assert condition_context_finetuned
        clip_context_ftmodel = CLIPModel.from_pretrained("./clip-vit-base-patch32")
        clip_context_ftprocessor = CLIPProcessor.from_pretrained("./clip-vit-base-patch32") 
        clip_context_ftmodel = clip_context_ftmodel.cuda()
        # LOAD
        clip_context_ftmodel.load_state_dict(torch.load(config.checkpoint_context_clip_projection), strict=False)
        # END LOAD
        clip_context_ftmodel.eval()
        
        for p in clip_context_ftmodel.parameters():
            p.requires_grad = False

        clip_context_ftmean = torch.Tensor([0.48145466, 0.4578275, 0.40821073]).to("cuda").unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        clip_context_ftstd = torch.Tensor([0.26862954, 0.26130258, 0.27577711]).to("cuda").unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        f_images2clip = lambda x: (torch.nn.functional.interpolate(x, [224, 224], mode = 'bilinear') - clip_context_ftmean) / clip_context_ftstd
        
        def clip_context_ftinference(imgs, txts):
            inputs = clip_context_ftprocessor(text=txts, return_tensors="pt", padding=True, truncation=True, max_length=77)
            inputs["input_ids"] = inputs["input_ids"].to("cuda")
            inputs["attention_mask"] = inputs["attention_mask"].to("cuda")
            inputs["pixel_values"] = f_images2clip(imgs)
            result = clip_context_ftmodel(**inputs)
            return result
        
        
    loss_optional_kwargs = {'label_clip_model' : label_clip_model, 
                            'label_processor' : label_processor, 
                            'prompt_path' : config.prompt_path,
                            'prompt_clip_model' : prompt_clip_model, 
                            'prompt_processor' : prompt_processor,
                            'clip_inference' : clip_inference,
                            'trainable_clip_inference' : trainable_clip_inference,
                            'clip_content_ftinference' : clip_content_ftinference,
                            'clip_context_ftinference' : clip_context_ftinference,
                            }
    
    loss_objs = [myloss.LOSS_NAME_TO_LOSS[_loss_name](weight = _loss_weight, **loss_optional_kwargs) \
        for _loss_name, _loss_weight in \
            zip(loss_names,
                loss_weights)]
    
    loss_names = rename_duplicates(loss_names)
    
    if trainable_clip_inference:
        optimizer = torch.optim.Adam(list(net.parameters()) + list(trainable_clip_model.parameters()), lr=config.lr, weight_decay=config.weight_decay)
        print("Optimizing net & trainable CLIP model parameters")
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        print("Optimizing net parameters")
    
    idx = 0
    init_epoch = 0
    if config.resume:
        init_epoch = int(os.path.split(config.resume)[-1].split(".")[0].split("Epoch")[-1]) 
        idx = init_epoch * len(train_loader)
    
    for epoch in range(init_epoch, config.num_epochs):
        print(f"Epoch {epoch}, iterations left: {len(train_loader)} ")
        for iteration, batch_data in enumerate(train_loader):
            
            total_loss = 0
            
            batch_data['low']  = batch_data['imgs'][0]
            batch_data['low'] = batch_data['low'].cuda()
            
            if config.dataloader_name == 'tiff_rgb_json_lowlight_loader':
                batch_data['jpg']  = batch_data['imgs'][1]
                batch_data['jpg'] = batch_data['jpg'].cuda()
        
            forward_func = lambda _net: getattr(_net, 'full_forward', net)

            net_out  = forward_func(net)(batch_data['low'])
            
            if isinstance(net_out, dict):
                assert NET_ENHANCED_KEY in net_out.keys()
                for _key in net_out.keys():
                    batch_data[_key] = net_out[_key]   
            else:
                assert isinstance(net_out, torch.Tensor)
                batch_data[NET_ENHANCED_KEY] = net_out
            
            for loss_obj, loss_name, loss_input in \
                zip(loss_objs, loss_names, loss_inputs):
                loss_val = loss_obj( *[batch_data[_key] for _key in loss_input] )
                writer.add_scalar(f'loss/loss_{loss_name}', loss_val, idx)
                total_loss += loss_val
                
            writer.add_scalar('loss/loss', total_loss , idx)

            idx += 1
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm(net.parameters(),config.grad_clip_norm)
            optimizer.step()

            if ((iteration+1) % config.display_iter) == 0:
                print("Loss at iteration", iteration+1, ":", total_loss.item())
            if ((iteration+1) % config.snapshot_iter) == 0:
                torch.save(net.state_dict(), os.path.join(experiment_dir, "Epoch" + str(epoch) + '.pth')) 		
        torch.save(net.state_dict(), os.path.join(experiment_dir, "Epoch" + str(epoch) + '.pth')) 		



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--lowlight_images_path', type=str, default="data/train_data/")
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--snapshots_folder', type=str, default="open_vocab_experiments/")
    parser.add_argument('--model', required=True, choices = model.MODEL_NAME_TO_MODEL.keys())
    parser.add_argument('--augmentation', required=True)
    parser.add_argument('--number_f', type=int, default=32)
    parser.add_argument('--name', type=str)
    parser.add_argument('--crop_size', type=int, default=256)

    parser.add_argument('--checkpoint', type=str, help='Pretrained checkpoint')
    parser.add_argument('--prompt_path', required=False, default="")
    parser.add_argument('--prompt_weight', type=float, default=1.0)

    parser.add_argument('--dataloader_name', type=str, required=True)
    parser.add_argument('--bbox_json', type=str, default=None)
    
    parser.add_argument('--checkpoint_content_clip_projection', type=str, default=None)
    parser.add_argument('--checkpoint_context_clip_projection', type=str, default=None)
    
    parser.add_argument('--loss_names', type=str, help='Separated by {PARS_SEPARATOR}', required=True)
    parser.add_argument('--loss_weights', type=str, help='Separated by {PARS_SEPARATOR}', required=True)
    parser.add_argument('--loss_inputs', type=str, help='Separated by {PARS_SEPARATOR} for each loss and {PARS_SEPARATOR2}', required=True)
    
    parser.add_argument('--resume', type=str, default=None, required=False)

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)

    train(config)

