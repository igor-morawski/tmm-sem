import os
import sys
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image, ImageOps
# import glob
from pathlib import Path
import random
import json
import torchvision
import cv2
random.seed(11)

HALF_PROB = 0.5 
EXTENSIONS = ["jpg", "png", "jpeg", "tiff", "JPG", "PNG", "JPEG", "TIFF"]

def populate_train_list(lowlight_images_path, shuffle=True, json_bboxes=None):
    extensions = [f".{ext}" for ext in EXTENSIONS]
    image_list_lowlight = [os.path.normpath(x) for x in Path(lowlight_images_path).iterdir() if x.suffix.lower() in extensions]    

    filename_list = [os.path.split(p)[-1] for p in image_list_lowlight]
    
    train_list = []
    if json_bboxes:
        for anno in json_bboxes["bboxes"]:
            train_list.append( (anno["filename"], anno) )
            assert anno["filename"] in filename_list
    else:
        for filename in filename_list:
            train_list.append( (filename, None) )
            
    if shuffle:
        random.shuffle(train_list)
    return train_list
    
class json_lowlight_loader(data.Dataset):
    AUGMENTATION_METHODS = ['resize', 'crop', 'crop+resize', 'rotate', 'crop+resize,rotate', 'hmirror', 'vmirror', 'ratioresize', 'zeropad2size', 'mirrorpad2size', 'none']

    def __init__(self, datasets_roots, json_bboxes_fp = None, shuffle = True, augmentation_method = 'resize', crop_size = 256, skip_validation=False, **kwargs):
        if kwargs: print(f"Skipping named arguments: {kwargs}")
        self.augmentation_method = augmentation_method
        for aug in self.augmentation_method.split(","):
            if not aug in self.AUGMENTATION_METHODS:
                raise Exception(f"{self.augmentation_method} not in supported methods ({self.AUGMENTATION_METHODS})")
        
        if not isinstance(datasets_roots, (list, tuple)):
            datasets_roots = [datasets_roots, ]
        self.datasets_roots = datasets_roots
        
        self.shuffle = shuffle
        self.size = crop_size
        
        self.json_bboxes_fp = json_bboxes_fp
        self.json_bboxes = None 
        if json_bboxes_fp:
            with open(self.json_bboxes_fp, "r") as json_file:
                self.json_bboxes = json.load(json_file)
                
        self.data_list = populate_train_list(self.datasets_roots[0], shuffle=self.shuffle, json_bboxes=self.json_bboxes) 
        
        if not skip_validation: 
            print("Validating dataset...") #XXX
            for filename, _ in self.data_list:
                for filepath in self._get_corresponding_filepaths(filename):
                    if not os.path.exists(filepath):
                        raise Exception(f"{filepath} does not exist")
            print("Dataset validation: OK")

        print("Total training examples:", len(self.data_list))
        
    def _get_corresponding_filepaths(self, filename, ):
        result = []
        for dataset_root in self.datasets_roots:
            result.append(os.path.join(dataset_root, filename))
        return result
        
    def __getitem__(self, index):
        filename = self.data_list[index][0]
        anno = self.data_list[index][1]
        filepaths = self._get_corresponding_filepaths(filename)
        
        if 'tiff' in filename.lower():
            imgs = [cv2.imread(filepath, cv2.IMREAD_UNCHANGED) for filepath in filepaths]
            for img in imgs:
                _, _, ch = img.shape
                assert ch == 3
                _unfolded = []
                for img in imgs:
                    _unfolded.append(img[:,:,0])
                    _unfolded.append(img[:,:,1])
                    _unfolded.append(img[:,:,2])
                imgs = _unfolded
                imgs = [Image.fromarray(img, mode='F') for img in imgs]
        else:
            imgs = [Image.open(filepath).convert('RGB') for filepath in filepaths]
        
        assert len(set([img.size for img in imgs]))
        assert all(os.path.exists(filepath) for filepath in filepaths)
    
        if anno: 
            content_xyxy = anno['content_xyxy']
            imgs = [img.crop(content_xyxy) for img in imgs]
            assert len(set([img.size for img in imgs]))
        
        # resize
        def _aug_resize(imgs):
            imgs = [img.resize((self.size,self.size), Image.LANCZOS) for img in imgs]
            x0, y0, x1, y1 = 0, 0, imgs[0].width, imgs[0].height
            return imgs
        
        # ratioresize
        def _aug_ratioresize(imgs):
            w, h = imgs[0].width, imgs[0].height
            max_dim = max(h, w)
            if max_dim > self.size:            
                scale = self.size / max_dim
                new_w, new_h = int(scale*w), int(scale*h)
                imgs = [img.resize((new_w, new_h), Image.LANCZOS) for img in imgs]
            return imgs
        
        
        def _aug_zeropad2size(imgs):
            width, height = imgs[0].size
            left, top = 0, 0
            bottom = (self.size - height % self.size) % self.size
            right = (self.size - width% self.size) % self.size
            if bottom == 0 and right == 0:
                return imgs
            new_width = width + right + left
            new_height = height + top + bottom
            assert new_width == self.size
            assert new_height == self.size
            results = [Image.new(img.mode, (new_width, new_height), color=0) for img in imgs]
            for result, img in zip(results, imgs):
                result.paste(img, (left, top))
            return results 
        
        def _aug_mirrorpad2size(imgs):
            width, height = imgs[0].size
            left, top = 0, 0
            bottom = (self.size - height % self.size) % self.size
            right = (self.size - width% self.size) % self.size
            if bottom == 0 and right == 0:
                return imgs
            new_width = width + right + left
            new_height = height + top + bottom
            assert new_width == self.size
            assert new_height == self.size
            arrays = [np.asarray(img) for img in imgs]
            arr_d_len = len(arrays[0].shape)
            if arr_d_len == 2:
                pads = ((0, bottom), (0, right), )
            elif arr_d_len == 3:
                pads = ((0, bottom), (0, right), (0, 0))
            else:
                raise Exception(arr_d_len)
            arrays = [np.pad(array, pads, mode='symmetric') for array in arrays]
            imgs = [Image.fromarray(array, imgs[0].mode) for array in arrays]
            for img in imgs:
                assert img.size[0] == self.size
                assert img.size[1] == self.size
            return imgs 
        
        # crop
        def _aug_crop(imgs):
            if (imgs[0].width <= self.size) or (imgs[0].height <= self.size):
                x0, y0, x1, y1 = 0, 0, imgs[0].width, imgs[0].height
                imgs = [img.resize((self.size,self.size), Image.LANCZOS) for img in imgs]
            else:
                x = random.randint(0, imgs[0].width - self.size)
                y = random.randint(0, imgs[0].height - self.size)
                x0, y0, x1, y1 = x, y, x + self.size, y + self.size
                imgs = [img.crop((x0, y0, x1, y1)) for img in imgs]
            return imgs
        
        # crop+resize 
        def _aug_crop_resize(imgs):
            if (imgs[0].width <= self.size) or (imgs[0].height <= self.size):
                imgs = [img.resize((self.size,self.size), Image.LANCZOS) for img in imgs]
                x0, y0, x1, y1 = 0, 0, imgs[0].width, imgs[0].height
            else:
                random_crop_size = random.randint(self.size, min(imgs[0].width, imgs[0].height))
                x = random.randint(0, imgs[0].width - random_crop_size)
                y = random.randint(0, imgs[0].height - random_crop_size)
                x0, y0, x1, y1 = x, y, x + random_crop_size, random_crop_size
                imgs = [img.crop((x0, y0, x1, y1)) for img in imgs]
                imgs = [img.resize((self.size,self.size), Image.LANCZOS) for img in imgs]
            return imgs
        
        # rotate
        def _aug_rotate(imgs):
            angle = random.choice([90, 180, 270]) # XXX
            imgs = [img.rotate(angle, expand=1) for img in imgs] 
            return imgs
        
        
        def _none(imgs):
            return imgs
        
        # hmirror
        def _aug_hmirror(imgs):
            prob = random.uniform(0, 1) 
            if prob <= HALF_PROB:
                imgs = [ImageOps.mirror(img) for img in imgs] 
            return imgs
        
        # vmirror
        def _aug_vmirror(imgs):
            prob = random.uniform(0, 1) 
            if prob <= HALF_PROB:
                imgs = [ImageOps.flip(img) for img in imgs] 
            return imgs

        aug_name2func = {
            'resize' : _aug_resize,
            'crop' : _aug_crop,
            'crop+resize' : _aug_crop_resize,
            'rotate' : _aug_rotate,
            'hmirror' : _aug_hmirror,
            'vmirror' : _aug_vmirror,
            'ratioresize' : _aug_ratioresize,
            'zeropad2size' : _aug_zeropad2size,
            'mirrorpad2size' : _aug_mirrorpad2size,
            'none' : _none,
        }
        pipeline = [aug_name2func[a] for a in self.augmentation_method.split(",")]
        
        def process(imgs, pipeline):
            for p in pipeline:
                imgs = p(imgs) 
            return imgs
        
        imgs = process(imgs, pipeline)
        
        assert imgs
        
        if 'tiff' in filename.lower():
            _folded = []
            for img_idx in range(0, len(imgs), 3):
                _folded.append(np.stack([np.asarray(imgs[img_idx]), np.asarray(imgs[img_idx+1]), np.asarray(imgs[img_idx+2])], -1))
            imgs = _folded
        else:
            imgs = [(np.asarray(img)/255.0) for img in imgs]
        imgs = [np.clip(img, 0, 1) for img in imgs]

        # assert all([img.shape == (self.size, self.size, 3) for img in imgs])
        
        imgs = [torch.from_numpy(img).float() for img in imgs]
        imgs = [img.permute(2, 0, 1) for img in imgs]
        
        result = {
                'imgs' : imgs, 
                'filepaths' : filepaths, 
                'filename' : filename,
                }
        
        if anno: assert self.json_bboxes_fp
        if anno:
            result['anno'] = anno
            result['content'] = result['anno']['content']
            result['context'] = result['anno']['context']
        return result

    def __len__(self):
        return len(self.data_list)


def array_pad_to_divisble_by(img, size):
    h, w, c = img.shape
    pad_h = (size - h % size) % size
    pad_w = (size - w % size) % size
    if pad_h == 0 and pad_w == 0:
        return img

    # Pad the image with mirrored pixels on the right and bottom edges
    pad_img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)

    return pad_img

def tensor_pad_to_divisble_by(img, size):
    b, c, h, w = img.size()
    pad_h = (size - h % size) % size
    pad_w = (size - w % size) % size
    if pad_h == 0 and pad_w == 0:
        return img

    # Pad the image with mirrored pixels on the right and bottom edges
    pad_img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode = 'reflect')

    return pad_img



# class tiff_rgb_json_lowlight_loader(json_lowlight_loader):
#     def __init__(self, *args, **kwargs):
#         super(tiff_rgb_json_lowlight_loader, self).__init__(*args, **kwargs)
#         print("Using coupled TIFF, RGB loader, validating that input roots are TIFF, RGB loaders")
#         if not len(self.datasets_roots) == 2:
#             raise Exception("Dataset roots are not TIFF, RGB")

if __name__ == "__main__":
    import cv2
    
    for j, j_name in zip([None, "/tmp2/igor/CM/data/generated_bboxes_score_20_overlapping.json"], ["def","bbox"]): 
        for a in ['ratioresize,zeropad2size', 'ratioresize,mirrorpad2size']:
            d = lowlight_loader(['/tmp2/igor/LL_Datasets/LOL/v1/our485/low', '/tmp2/igor/LL_Datasets/LOL/v1/our485/high'],json_bboxes_fp=j,augmentation_method=a, shuffle=False)
            imgs = d[1]['imgs']
            for idx, i in enumerate(imgs):
                i = i.permute(1, 2, 0).numpy()
                fp = f"{j_name}_{idx}_{a.replace(',', '_')}.png"
                cv2.imwrite(fp, i[...,::-1]*255,)
    