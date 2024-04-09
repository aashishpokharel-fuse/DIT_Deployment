import numpy as np
import torch
import os
import numpy as np
import torch
import torch.multiprocessing
from PIL import Image, ImageOps, ImageFilter
import gc
import yaml
import re

from ocr.custom_ocr.config import OCRConfig
from ocr.custom_ocr.model_block import CharModel
from ocr.custom_ocr.ocr_model import OCR_Model
from ocr.custom_ocr.utils import read_vocab_file

# CONFIG_PATH = "./"
CONFIG_PATH = "./"

def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)

    return config


config = load_config("config.yaml")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

char_model = CharModel()
char_model(read_vocab_file(config['custom_ocr_char_path']))

ocr_config = OCRConfig().from_json_file(config['custom_ocr_config_path'])
custom_ocr_batch_size = config['custom_ocr_batch_size']

ocr_model = OCR_Model(char_model, ocr_config, device, custom_ocr_batch_size)
checkpoint = torch.load(config['custom_ocr_path'], map_location=f"{device}")
ocr_model = ocr_model.module if hasattr(ocr_model, "module") else ocr_model
ocr_model.load_state_dict(checkpoint["model_state_dict"])
ocr_model.eval()
ocr_model.to(device)


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

def predict_char_seq(preprocessed_images, model=ocr_model):
    with torch.no_grad():
        text_lists = []
        text_confidence = []
        batch_size = model.batch_size
        for i in range(0, len(preprocessed_images), batch_size):
            batch_imgs = preprocessed_images[i:i+batch_size]
            batch_data = [np.expand_dims(test_img, axis=0).repeat(3, axis=0) / 255.0 for test_img in batch_imgs]
            batch_data = np.stack(batch_data, axis=0).astype(np.float32)
            
            img2txt_dec_txt, confidence = model.generate(batch_data, search_type="greedy")
            text_lists.extend(post_process(img2txt_dec_txt, model.char_model))
            text_confidence.extend(confidence)

        return text_lists, text_confidence

def post_process(predicted, char_model, raw_prob=False, ctc_mode=False):
    raw_texts = []
    with torch.no_grad():
        if raw_prob:
            predicted = torch.argmax(predicted, dim=-1)
        bs = len(predicted)
        token = "TEOS"
        for i in range(bs):
            str_predicted = (char_model.indexes2characters(predicted[i].cpu().numpy(), ctc_mode))
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
            
            str_predicted = str(str_predicted)
            str_predicted = re.sub(" +", " ", str_predicted)
            str_predicted = re.sub('’', "'", str_predicted)

            if len(str_predicted.strip()) != 0:
                str_predicted = str_predicted.strip()
            
            ours_is_digit = [True if d in ["(",")",".",",","0","1","2","3","4","5","6","7","8","9"] else False for d in str_predicted]
            if all(ours_is_digit):
                len_ours = len(str_predicted)
                new_ours = ""
                for index, c in enumerate(str_predicted):
                    # print(len_ours-index, c)
                    if len_ours - index > 5 and c==".":
                        new_ours += ","
                    else:
                        new_ours += c
                str_predicted = new_ours

            if all(char.isdigit() or char in [",","(",")","."] for char in str_predicted):
                    str_predicted = str_predicted.replace(',', '')      

            raw_texts.append(str_predicted)
            
        return raw_texts


def predict(images, image_height=64, image_width=1536):
    # try:
        preprocessed_images = []
        predicted_texts = []
        
        for image in images:
            file_index = 0
            # while os.path.exists(f"./output/ocr_data/temp_{file_index}.jpg"):
            #     file_index += 1
            # image.save(f"./output/ocr_data/temp_{file_index}.jpg")
            img = preprocess_image(image, image_height, image_width)
            preprocessed_images.append(img)
        
        predicted_texts, confidence = predict_char_seq(preprocessed_images)
        gc.collect()
        return predicted_texts, confidence
    
    # except Exception as e:
    #     error_message = str(e)
    #     raise Exception(error_message)
