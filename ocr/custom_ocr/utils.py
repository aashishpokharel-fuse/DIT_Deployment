import numpy as np
import torch
import os
import time as t
import numpy as np
import torch
import gc
import torch.multiprocessing
from PIL import Image, ImageOps, ImageFilter
from torch.nn import functional as f
import cv2
import re

def get_mask_seq_cat_np(first_seq_len=128, second_seq_len=128):
    second_mask = np.triu(np.full((second_seq_len, second_seq_len), float("-inf")), k=1)
    first_mask = np.zeros((second_seq_len, first_seq_len))

    bottom_mask = np.concatenate([first_mask, second_mask], axis=-1)
    top_mask = np.tile(bottom_mask[0], (first_seq_len, 1))

    mask = np.concatenate([top_mask, bottom_mask], axis=0)
    return mask

def get_mask_seq_cat(first_seq_len=128, second_seq_len=128):
    second_mask = torch.triu(torch.full((second_seq_len, second_seq_len), float("-inf")), diagonal=1)
    first_mask = torch.zeros(second_seq_len, first_seq_len)

    bottom_mask = torch.cat([first_mask, second_mask], axis=-1)
    top_mask = bottom_mask[0].unsqueeze(0).repeat(first_seq_len,1)

    mask = torch.cat([top_mask, bottom_mask], axis=0)
    return mask

def read_vocab_file(vocab_file):
    with open(vocab_file, "r") as f:
        collection = f.read().splitlines()

    return collection

def preprocess_image(image, CIH, CIW):
    img = ImageOps.grayscale(image)
    img = resize_img(img, h=CIH, w=CIW)
    
    kernel = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])

    img = img.filter(ImageFilter.Kernel((3, 3), kernel.flatten(), scale=1))
    img = np.array(img)

    return img

def resize_img(img, h=64, w=1024, uniform_shape=True, fill=0):
    IMAGE_WIDTH = w
    IMAGE_HEIGHT = h
    img_dim = (IMAGE_WIDTH, IMAGE_HEIGHT)

    if uniform_shape:
        img_w, img_h = img.size
        max_w, max_h = img_dim
        coef = max(img_h / max_h, img_w / max_w)
        h = int(img_h / coef)
        w = int(img_w / coef)
        img = img.resize((w, h))
        pad_h, pad_w = max_h - h, max_w - w
        if pad_h > 0:
            pad_pixels = pad_h
            up_pad = int(pad_pixels/2)
            down_pad = pad_pixels - up_pad
            padding = (0, up_pad, 0, down_pad)
            # padding = (0, 0, 0, pad_h)
            img = ImageOps.expand(img, padding, fill=fill)
        if pad_w > 0:
            padding = (0, 0, pad_w, 0)
            img = ImageOps.expand(img, padding, fill=fill)
    else:
        if img.size[0] > IMAGE_WIDTH:
            img = img.resize(img_dim)
        else:
            padding = (0, 0, IMAGE_WIDTH-img.size[0], fill)
            img = ImageOps.expand(img, padding, fill=fill)

    return img

def post_process(predicted, char_model, raw_prob=False, ctc_mode=False):
    raw_texts = []
    with torch.no_grad():
        if raw_prob:
            predicted = torch.argmax(predicted, dim=-1)
        bs = predicted.shape[0]
        token = "TEOS"
        for i in range(bs):
            str_predicted = (char_model.indexes2characters(predicted[i].cpu().numpy(), ctc_mode))
            # print(str_predicted)
            if token in str_predicted:
                str_predicted_first_pad_index = str_predicted.index(token)
            else:
                if "PAD" in str_predicted:
                    str_predicted_first_pad_index = str_predicted.index("PAD")
                else:
                    str_predicted_first_pad_index = len(str_predicted)
            str_predicted = "".join(str_predicted[:str_predicted_first_pad_index])

            if str_predicted.startswith("TSOS"):
                str_predicted = str_predicted[4:]

            raw_texts.append(str_predicted)
            
        return raw_texts
