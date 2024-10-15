import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16


class LCLIP_img_txt(nn.Module):
    def __init__(self, weight = 1, clip_inference = None, **kwargs):
        super(LCLIP_img_txt, self).__init__()
        assert clip_inference
        self.clip_inference = clip_inference
        self.weight = weight
        self.img_ce = nn.CrossEntropyLoss(reduction='mean')
        self.txt_ce = nn.CrossEntropyLoss(reduction='mean')
        
    def forward(self, imgs, txts):
        clip_result = self.clip_inference(imgs, txts)        
        ground_truth = torch.arange(len(imgs),dtype=torch.long,device=imgs[0].device)
        logits_per_image = clip_result['logits_per_image']
        logits_per_text = clip_result['logits_per_text']
        loss_val = (self.img_ce(logits_per_image,ground_truth) + self.txt_ce(logits_per_text,ground_truth)) / 2
        return self.weight * loss_val


class L_trainable_CLIP_img_txt(nn.Module):
    def __init__(self, weight = 1, trainable_clip_inference = None, **kwargs):
        super(L_trainable_CLIP_img_txt, self).__init__()
        assert trainable_clip_inference
        self.trainable_clip_inference = trainable_clip_inference
        self.weight = weight
        self.img_ce = nn.CrossEntropyLoss(reduction='mean')
        self.txt_ce = nn.CrossEntropyLoss(reduction='mean')
        
    def forward(self, imgs, txts):
        clip_result = self.trainable_clip_inference(imgs, txts)        
        ground_truth = torch.arange(len(imgs),dtype=torch.long,device=imgs[0].device)
        logits_per_image = clip_result['logits_per_image']
        logits_per_text = clip_result['logits_per_text']
        loss_val = (self.img_ce(logits_per_image,ground_truth) + self.txt_ce(logits_per_text,ground_truth)) / 2
        return self.weight * loss_val




class L_trainable_CLIP_img(nn.Module):
    def __init__(self, weight = 1, trainable_clip_inference = None, **kwargs):
        super(L_trainable_CLIP_img, self).__init__()
        assert trainable_clip_inference
        self.trainable_clip_inference = trainable_clip_inference
        self.weight = weight
        self.img_ce = nn.CrossEntropyLoss(reduction='mean')
        
    def forward(self, imgs, txts):
        clip_result = self.trainable_clip_inference(imgs, txts)        
        ground_truth = torch.arange(len(imgs),dtype=torch.long,device=imgs[0].device)
        logits_per_image = clip_result['logits_per_image']
        loss_val = self.img_ce(logits_per_image,ground_truth)
        return self.weight * loss_val


class LCLIP_img_txt_contr_eh(nn.Module):
    def __init__(self, weight = 1, clip_inference = None, **kwargs):
        super(LCLIP_img_txt_contr_eh, self).__init__()
        assert clip_inference
        self.clip_inference = clip_inference
        self.weight = weight
        self.img_ce = nn.CrossEntropyLoss(reduction='mean')
        self.txt_ce = nn.CrossEntropyLoss(reduction='mean')
        self.mod_ce = nn.CrossEntropyLoss(reduction='mean')
        
    def forward(self, imgs_enhanced, imgs_high, txts):
        clip_result_enh = self.clip_inference(imgs_enhanced, txts)        
        logits_per_image_enh = clip_result_enh['logits_per_image']
        diag_enh = torch.diagonal(logits_per_image_enh)
        
        clip_result_hi = self.clip_inference(imgs_high, txts)        
        logits_per_image_hi = clip_result_hi['logits_per_image']
        diag_hi = torch.diagonal(logits_per_image_hi)
        
        ground_truth = torch.zeros(len(imgs_enhanced),dtype=torch.long,device=imgs_enhanced[0].device)
        
        diags = torch.stack([diag_enh, diag_hi], ).transpose(1, 0)
        logits_per_image = clip_result_enh['logits_per_image']
        logits_per_text = clip_result_enh['logits_per_text']
        loss_val = (self.img_ce(logits_per_image,ground_truth) + self.txt_ce(logits_per_text,ground_truth) + self.mod_ce(diags, ground_truth)) / 3 
        return self.weight * loss_val


class LCLIP_img(nn.Module):
    def __init__(self, weight = 1, clip_inference = None, **kwargs):
        super(LCLIP_img, self).__init__()
        assert clip_inference
        self.clip_inference = clip_inference
        self.weight = weight
        self.img_ce = nn.CrossEntropyLoss(reduction='mean')
        
    def forward(self, imgs, txts):
        clip_result = self.clip_inference(imgs, txts)        
        ground_truth = torch.arange(len(imgs),dtype=torch.long,device=imgs[0].device)
        logits_per_image = clip_result['logits_per_image']
        loss_val = self.img_ce(logits_per_image,ground_truth)
        return self.weight * loss_val
    
    
    
    
    
    
    
    
    

class LCLIP_img_txt_finetuned_content(nn.Module):
    def __init__(self, weight = 1, clip_content_ftinference = None, **kwargs):
        super(LCLIP_img_txt_finetuned_content, self).__init__()
        assert clip_content_ftinference
        self.clip_inference = clip_content_ftinference
        self.weight = weight
        self.img_ce = nn.CrossEntropyLoss(reduction='mean')
        self.txt_ce = nn.CrossEntropyLoss(reduction='mean')
        
    def forward(self, imgs, txts):
        clip_result = self.clip_inference(imgs, txts)        
        ground_truth = torch.arange(len(imgs),dtype=torch.long,device=imgs[0].device)
        logits_per_image = clip_result['logits_per_image']
        logits_per_text = clip_result['logits_per_text']
        loss_val = (self.img_ce(logits_per_image,ground_truth) + self.txt_ce(logits_per_text,ground_truth)) / 2
        return self.weight * loss_val
    
    

class LCLIP_img_txt_finetuned_context(nn.Module):
    def __init__(self, weight = 1, clip_context_ftinference = None, **kwargs):
        super(LCLIP_img_txt_finetuned_context, self).__init__()
        assert clip_context_ftinference
        self.clip_inference = clip_context_ftinference
        self.weight = weight
        self.img_ce = nn.CrossEntropyLoss(reduction='mean')
        
    def forward(self, imgs, txts):
        clip_result = self.clip_inference(imgs, txts)        
        ground_truth = torch.arange(len(imgs),dtype=torch.long,device=imgs[0].device)
        logits_per_image = clip_result['logits_per_image']
        loss_val = self.img_ce(logits_per_image,ground_truth)
        return self.weight * loss_val
    
 
 
class LCLIP_img_finetuned_context(nn.Module):
    def __init__(self, weight = 1, clip_context_ftinference = None, **kwargs):
        super(LCLIP_img_finetuned_context, self).__init__()
        assert clip_context_ftinference
        self.clip_inference = clip_context_ftinference
        self.weight = weight
        self.img_ce = nn.CrossEntropyLoss(reduction='mean')
        self.txt_ce = nn.CrossEntropyLoss(reduction='mean')
        
    def forward(self, imgs, txts):
        clip_result = self.clip_inference(imgs, txts)        
        ground_truth = torch.arange(len(imgs),dtype=torch.long,device=imgs[0].device)
        logits_per_image = clip_result['logits_per_image']
        logits_per_text = clip_result['logits_per_text']
        loss_val = (self.img_ce(logits_per_image,ground_truth) + self.txt_ce(logits_per_text,ground_truth)) / 2
        return self.weight * loss_val
    
    

class LCLIP_img_finetuned_content(nn.Module):
    def __init__(self, weight = 1, clip_content_ftinference = None, **kwargs):
        super(LCLIP_img_finetuned_content, self).__init__()
        assert clip_content_ftinference
        self.clip_inference = clip_content_ftinference
        self.weight = weight
        self.img_ce = nn.CrossEntropyLoss(reduction='mean')
        
    def forward(self, imgs, txts):
        clip_result = self.clip_inference(imgs, txts)        
        ground_truth = torch.arange(len(imgs),dtype=torch.long,device=imgs[0].device)
        logits_per_image = clip_result['logits_per_image']
        loss_val = self.img_ce(logits_per_image,ground_truth)
        return self.weight * loss_val
    
       