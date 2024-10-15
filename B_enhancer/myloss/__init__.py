import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .zdce import L_color,L_spa,L_exp,L_TV
from .clip_losses import L_CLIP_BCE_Label, L_CLIP_Pretrained_Prompt
from .contextual import LCLIP_img_txt, L_trainable_CLIP_img_txt, L_trainable_CLIP_img, LCLIP_img_txt_contr_eh, LCLIP_img
from .contextual import LCLIP_img_txt_finetuned_content, LCLIP_img_txt_finetuned_context, LCLIP_img_finetuned_context, LCLIP_img_finetuned_content


class L1(nn.Module):
    def __init__(self, weight = 1, **kwargs):
        super(L1, self).__init__()
        self._func = torch.nn.functional.l1_loss
        self.weight = weight
        
    def forward(self, x, y):
        return self.weight * self._func(x, y).mean()

class L2(nn.Module):
    def __init__(self, weight = 1, **kwargs):
        super(L2, self).__init__()
        self._func = torch.nn.functional.l2_loss
        self.weight = weight
        
    def forward(self, x, y):
        return self.weight * self._func(x, y).mean()






LOSS_NAME_TO_LOSS = {
                       'L1' : L1,
                       'L2' : L2,
                       'L_color' : L_color,
                       'L_spa' : L_spa,
                       'L_exp' : L_exp,
                       'L_TV' : L_TV, 
                       'L_CLIP_BCE_Label' : L_CLIP_BCE_Label,
                       'L_CLIP_Pretrained_Prompt' : L_CLIP_Pretrained_Prompt,
                       'LCLIP_img_txt' : LCLIP_img_txt,
                       'L_trainable_CLIP_img_txt' : L_trainable_CLIP_img_txt,
                       'LCLIP_img_txt_contr_eh' : LCLIP_img_txt_contr_eh,
                       'LCLIP_img' : LCLIP_img,
                       'L_trainable_CLIP_img' : L_trainable_CLIP_img,
                       'LCLIP_img_txt_finetuned_content' : LCLIP_img_txt_finetuned_content,
                       'LCLIP_img_txt_finetuned_context' : LCLIP_img_txt_finetuned_context,
                       'LCLIP_img_finetuned_context' : LCLIP_img_finetuned_context,
                       'LCLIP_img_finetuned_content' : LCLIP_img_finetuned_content
                       }


