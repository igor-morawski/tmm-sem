import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from collections import OrderedDict

from CLIP import clip

def create_mask(x, num_classes):
    result  = torch.nn.functional.one_hot(2*x, num_classes = 2 * num_classes)
    result += torch.nn.functional.one_hot(2*x + 1, num_classes = 2 * num_classes)
    return result

def create_pos_mask(x, num_classes):
    result  = torch.nn.functional.one_hot(2*x, num_classes = 2 * num_classes)
    return result


class MaskedSoftmax(torch.nn.Module):
    def __init__(self):
        super(MaskedSoftmax, self).__init__()
        self.softmax = torch.nn.Softmax(1)

    def forward(self, x, mask=None):
        """
        Performs masked softmax, as simply masking post-softmax can be
        inaccurate
        :param x: [batch_size, num_items]
        :param mask: [batch_size, num_items]
        :return:
        """
        if x.size(1) == 2:
            return torch.nn.functional.softmax(x, dim=-1)
        x_masked = x.masked_fill(mask.to(torch.bool), -torch.inf)
        x_max = x_masked.max(1)[0]
        x_exp = (x - x_max.unsqueeze(-1)).exp()
        if mask is not None:
            x_exp = x_exp * mask.float()
        return x_exp / x_exp.sum(1).unsqueeze(-1)
    
POS_PROMPT_TEMPLATE = "a photo of a {}"
NEG_PROMPT_TEMPLATE = "not a photo of a {}"


class L_CLIP_BCE_Label(nn.Module):
    def __init__(self, weight = 1, label_clip_model = None, label_processor = None, **kwargs):
        super(L_CLIP_BCE_Label, self).__init__()
        assert label_clip_model
        assert label_processor
        self.weight = weight
        self.masked_softmax = MaskedSoftmax()
        self.L_BCE = torch.nn.BCELoss(reduce = False) 
        self.label_processor = label_processor 
        self.label_clip_model = label_clip_model
        
    def forward(self, enhanced, labels):
        classes = list(set(labels))
        labels = torch.Tensor([classes.index(_l) for _l in labels]).to(enhanced.device)
        levels = [1, ] 
        clip_mean = torch.Tensor([0.48145466, 0.4578275, 0.40821073]).to("cuda").unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        clip_std = torch.Tensor([0.26862954, 0.26130258, 0.27577711]).to("cuda").unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        assert levels == [1, ]
        out_levels = []
        for level in levels:
            text = [(POS_PROMPT_TEMPLATE.format(c),
                    NEG_PROMPT_TEMPLATE.format(c)) for c in classes]
            text = [item for sublist in text for item in sublist]
            mask = create_mask(torch.tensor(labels.squeeze()).to(torch.long), num_classes = len(classes))
            pos_mask = create_pos_mask(torch.tensor(labels.squeeze()).to(torch.long), num_classes = len(classes))
            f_images2clip = lambda x: (torch.nn.functional.interpolate(x, [224, 224], mode = 'bilinear') - clip_mean) / clip_std
            images4clip = f_images2clip(enhanced)            
            inputs = self.label_processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=77)
            inputs["input_ids"] = inputs["input_ids"].to("cuda")
            inputs["attention_mask"] = inputs["attention_mask"].to("cuda")
            inputs["pixel_values"] = images4clip
            outputs = self.label_clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            logits_per_image = torch.clip(logits_per_image, 1e-6, None)
            sm = self.masked_softmax(logits_per_image, mask)
            sm_ce = (sm * pos_mask).sum(axis=-1)
            l_bce = self.L_BCE(sm_ce, torch.ones_like(sm_ce).to(sm_ce.device))
            l_bce = self.weight * l_bce.mean()
            out_levels.append(l_bce)
        l_bce = sum(out_levels) / len(out_levels)

        return self.weight * l_bce
    
    
    
    
    
    

class TextEncoder(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.transformer = m.transformer
        self.positional_embedding = m.positional_embedding
        self.ln_final = m.ln_final
        self.text_projection = m.text_projection
        self.dtype = m.dtype

    def forward(self, prompts, tokenized_prompts):

        x = prompts + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        
        return x

class Prompts(nn.Module):
    def __init__(self,initials=None,m=None):
        super(Prompts,self).__init__()
        assert m is not None
        print("The initial prompts are:",initials)
        self.text_encoder = TextEncoder(m)
        if isinstance(initials,list):
            text = clip.tokenize(initials).cuda()
            self.embedding_prompt = nn.Parameter(m.token_embedding(text).requires_grad_()).cuda()
        elif isinstance(initials,str):
            prompt_path=initials

            state_dict = torch.load(prompt_path)
            # create new OrderedDict that does not contain `module.`
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                # name = k[7:] # remove `module.` # XXX
                name = k
                new_state_dict[name] = v
            self.embedding_prompt=nn.Parameter(new_state_dict['embedding_prompt']).cuda()
            self.embedding_prompt.requires_grad = True
        else:
            self.embedding_prompt=torch.nn.init.xavier_normal_(nn.Parameter(m.token_embedding([" ".join(["X"]*16)," ".join(["X"]*16)]).requires_grad_())).cuda()

    def forward(self,tensor,flag=1):
        tokenized_prompts= torch.cat([clip.tokenize(p) for p in [" ".join(["X"]*16)]])
        tokenized_prompts = tokenized_prompts.to(self.embedding_prompt.dtype)
        text_features = self.text_encoder(self.embedding_prompt,tokenized_prompts)
        
        for i in range(tensor.shape[0]):
            image_features=tensor[i]
            nor=torch.norm(text_features,dim=-1, keepdim=True)
            if flag==0:
                similarity = (100.0 * image_features.unsqueeze(0) @ (text_features/nor).T)#.softmax(dim=-1)
                if(i==0):
                    probs=similarity
                else:
                    probs=torch.cat([probs,similarity],dim=0)
            else:
                similarity = (100.0 * image_features @ (text_features/nor).T).softmax(dim=-1)#/nor
                if(i==0):
                    probs=similarity[:,0]
                else:
                    probs=torch.cat([probs,similarity[:,0]],dim=0)
        return probs
    
    
    
    
    
    
    
class L_CLIP_Pretrained_Prompt(nn.Module):
    def __init__(self, weight = 1, prompt_path = None, prompt_clip_model = None, prompt_processor = None, **kwargs):
        super(L_CLIP_Pretrained_Prompt, self).__init__()
        assert prompt_processor 
        assert prompt_clip_model
        self.weight = weight
        
        def encode_and_normalize(img_t):
            img_t = prompt_clip_model.encode_image(img_t)
            img_t = torch.div(img_t, img_t.norm(dim=-1, keepdim=True))
            return img_t
        self.encode_and_normalize = encode_and_normalize
            
        self.pretrained_prompt = Prompts(prompt_path, prompt_clip_model)
        
        
    def forward(self, enhanced, ):
        clip_mean = torch.Tensor([0.48145466, 0.4578275, 0.40821073]).to(enhanced.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1) 
        clip_std = torch.Tensor([0.26862954, 0.26130258, 0.27577711]).to(enhanced.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1) 
        f_images2clip = lambda x: (torch.nn.functional.interpolate(x, [224, 224], mode = 'bilinear') - clip_mean) / clip_std # XXX
        f_sm = lambda x: torch.nn.functional.softmax(x, dim=-1)
        prompt_logits = self.pretrained_prompt(self.encode_and_normalize(f_images2clip(enhanced)),0)
        prompt_loss = torch.nn.functional.cross_entropy(prompt_logits,
                                    torch.ones([enhanced.size(0), ]).to(enhanced.device).to(torch.long),)
        return self.weight * prompt_loss



