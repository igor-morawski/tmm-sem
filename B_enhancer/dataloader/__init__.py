from .json_loader import json_lowlight_loader
import cv2

def pad_to_divisble_by(img, size):
    h, w, c = img.shape
    pad_h = (size - h % size) % size
    pad_w = (size - w % size) % size
    if pad_h == 0 and pad_w == 0:
        return img

    # Pad the image with mirrored pixels on the right and bottom edges
    pad_img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)

    return pad_img

DATALOADER_NAME_TO_DATALOADER = {
                        'json_lowlight_loader' : json_lowlight_loader, 
                       }