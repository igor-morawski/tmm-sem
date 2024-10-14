import argparse

# import CLIP
from CLIP import clip

import os
from collections import OrderedDict
import json

import dataloader

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
import torch.optim

from torchvision.transforms import ColorJitter

from datetime import datetime
from tensorboardX import SummaryWriter



class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):

        x = prompts + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(
            dim=-1)] @ self.text_projection

        return x


class Prompts(nn.Module):
    def __init__(self, initials=None):
        super(Prompts, self).__init__()
        print("The initial prompts are:", initials)
        self.text_encoder = TextEncoder(model)
        if isinstance(initials, list):
            text = clip.tokenize(initials).cuda()
            self.embedding_prompt = nn.Parameter(
                model.token_embedding(text).requires_grad_()).cuda()
        elif isinstance(initials, str):
            prompt_path = initials

            state_dict = torch.load(prompt_path)
            # create new OrderedDict that does not contain `module.`
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            self.embedding_prompt = nn.Parameter(
                new_state_dict['embedding_prompt']).cuda()
            self.embedding_prompt.requires_grad = True
        else:
            self.embedding_prompt = torch.nn.init.xavier_normal_(nn.Parameter(model.token_embedding(
                [" ".join(["X"]*config.length_prompt), " ".join(["X"]*config.length_prompt)]).requires_grad_())).cuda()

    def forward(self, tensor, flag=1):
        tokenized_prompts = torch.cat(
            [clip.tokenize(p) for p in [" ".join(["X"]*config.length_prompt)]])
        tokenized_prompts = tokenized_prompts.to(self.embedding_prompt.dtype)
        text_features = self.text_encoder(
            self.embedding_prompt, tokenized_prompts)

        for i in range(tensor.shape[0]):
            image_features = tensor[i]
            nor = torch.norm(text_features, dim=-1, keepdim=True)
            if flag == 0:
                similarity = (100.0 * image_features.unsqueeze(0)
                              @ (text_features/nor).T)  # .softmax(dim=-1)
                if (i == 0):
                    probs = similarity
                else:
                    probs = torch.cat([probs, similarity], dim=0)
            else:
                similarity = (100.0 * image_features @
                              (text_features/nor).T).softmax(dim=-1)  # /nor
                if (i == 0):
                    probs = similarity[:, 0]
                else:
                    probs = torch.cat([probs, similarity[:, 0]], dim=0)
        return probs


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--length_prompt', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--prompt_lr', type=float, default=0.000005)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--snapshots_folder', type=str,
                        default="augmentation_prompts")
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--augmentation_method', type=str, default="crop")
    parser.add_argument('--crop_size', type=int, default=1024)
    parser.add_argument('--name', type=str, default="")
    parser.add_argument('--brightness_augmentation', action="store_true")

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)

    now = datetime.now()
    date_string = now.strftime("%y%m%d_%H%M%S")
    dir_name = f"prompts_{date_string}"
    if config.name:
        dir_name += config.name
    experiment_dir = os.path.join(config.snapshots_folder, dir_name)

    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    with open(os.path.join(experiment_dir, "args.json"), 'w') as f:
        json.dump(vars(config), f, indent=4)
    writer = SummaryWriter(os.path.join(experiment_dir, "logs"))

    model, preprocess = clip.load(
        "ViT-B/32", device=torch.device("cuda"), download_root="./clip_model/")  # ViT-B/32
    model = model.to(torch.float32)
    learn_prompt = Prompts([" ".join(
        ["X"]*(config.length_prompt)), " ".join(["X"]*(config.length_prompt))]).cuda()
    prompt_optimizer = torch.optim.Adam(learn_prompt.parameters(
    ), lr=config.prompt_lr, weight_decay=config.weight_decay)

    train_dataset = dataloader.lowlight_loader(lowlight_images_path=config.dataset_path,
                                               encoder=dataloader.Encoder(),
                                               augmentation_method=config.augmentation_method,
                                               crop_size=config.crop_size,
                                               skip_validation=True)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config.batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True)

    def identity(x): return x
    def subsample4x(x): return x[:, :, ::4, ::4]
    def downsample4x(x): return torch.nn.functional.avg_pool2d(x, 4, 4)

    def batch_tensor2np(x): return x.permute(0, 2, 3, 1, ).cpu().numpy()
    clip_mean = torch.Tensor([0.48145466, 0.4578275, 0.40821073]).to(
        "cuda").unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    clip_std = torch.Tensor([0.26862954, 0.26130258, 0.27577711]).to(
        "cuda").unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    def f_images2clip(x): return (torch.nn.functional.interpolate(
        x, [224, 224], mode='bilinear') - clip_mean) / clip_std

    if config.brightness_augmentation:
        cj = ColorJitter(brightness=(0.5, 2), contrast=(
            1), saturation=(0.5, 1.5), hue=(-0.1, 0.1))

    iteration_idx = 0
    print(len(train_loader))
    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch}")
        for _, img_lowlight in enumerate(train_loader):
            img_lowlight = img_lowlight.cuda()
            if config.brightness_augmentation:
                img_lowlight = cj(img_lowlight)
            B, C, H, W = img_lowlight.size()
            label0 = torch.zeros([B, ]).to(img_lowlight.device).to(torch.long)
            label1 = torch.ones([B, ]).to(img_lowlight.device).to(torch.long)
            sub_sample = subsample4x(img_lowlight)
            down_sample = downsample4x(img_lowlight)

            image_features0 = model.encode_image(f_images2clip(sub_sample))
            image_features0 = torch.div(
                image_features0, image_features0.norm(dim=-1, keepdim=True))
            output0 = learn_prompt(image_features0, 0)
            loss = F.cross_entropy(output0, label0).mean()

            image_features1 = model.encode_image(f_images2clip(down_sample))
            image_features1 = torch.div(
                image_features1, image_features1.norm(dim=-1, keepdim=True))
            output1 = learn_prompt(image_features1, 0)
            loss += F.cross_entropy(output1, label1).mean()

            writer.add_scalar('loss/Loss_CE', loss, iteration_idx)

            prompt_optimizer.zero_grad()
            loss.backward()
            prompt_optimizer.step()
            if not (iteration_idx % 10):
                print(f"{iteration_idx}: {loss.mean()}")
            iteration_idx += 1
        torch.save(learn_prompt.state_dict(), os.path.join(
            experiment_dir, "Epoch" + str(epoch) + '.pth'))
