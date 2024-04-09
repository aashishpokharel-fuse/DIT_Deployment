import fitz
import json
from PIL import Image
from io import BytesIO
import base64
import os
import time
from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np
import cv2
# import pytesseract
import re

from ocr.custom_ocr.inference import predict


def extract_lines_from_text_image(image, img_index):
    """
    Extracts lines from the given image.

    Args:
        image: The text image from which to extract lines.

    Returns:
        A list of extracted lines (regions of interest) from the image.
    """
    width, height = image.size

    gray_image = image.convert('L')
    blur = cv2.GaussianBlur(np.array(gray_image.copy()), (3, 3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # remove vertical line
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,25))
    t_img = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    t_img = thresh - t_img

    t_blur = cv2.GaussianBlur(t_img,(3,3),0)
    t_thresh = cv2.threshold(t_blur, 100, 255, cv2.THRESH_BINARY)[1]

    # remove horizontal line
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,1))
    t_img = cv2.morphologyEx(t_thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    t_img = t_thresh - t_img

    t_blur = cv2.GaussianBlur(t_img,(3,3),0)
    thresh = cv2.threshold(t_blur, 100, 255, cv2.THRESH_BINARY)[1]
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
    cntrs = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    lines = []
    if len(cntrs) > 1:
        for index, cnt in enumerate(cntrs):
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            x, y, w, h = cv2.boundingRect(cnt)
            x_start = max(min(x - 5, width), 0)
            x_end = max(min(x + w + 5, width), 0)
            y_start = max(min(y - 5, height), 0)
            y_end = max(min(y + h + 5, height), 0)
            if h >= 10:
                lines.append(image.crop((x_start,y_start,x_end,y_end)))
            
    else:
        return [image]
    
    return lines

def extract_text_by_boxes(image, bounding_box, exclude_index):
    """
    Extracts text from a given image within specified bounding boxes.

    Parameters:
    - image (PIL.Image): The image from which to extract text.
    - bounding_box (list): A list of bounding boxes within which to extract text. Each bounding box is a tuple of (left, top, right, bottom) coordinates.
    - exclude_index (list): A list of indices of bounding boxes to exclude from text extraction.

    Returns:
    - list: A list of extracted text from each bounding box. The text from each bounding box is a string. If no text is extracted from a bounding box, the corresponding list element is an empty string.
    """
    extracted_text = ["" for _ in range(len(bounding_box))]
    extracted_text_confidence = [0 for _ in range(len(bounding_box))]
    
    lines_count_dict = {}
    lines = []
    null_image_index = []
    
    for i, box in enumerate(bounding_box):

        if i not in exclude_index:
            left, top, right, bottom = box   
            rect = image.crop((left,top,right,bottom))
            h,w = rect.size
            if h==0 or w==0:
                null_image_index.append(i)
                continue

            segmented_lines = extract_lines_from_text_image(rect,i)
            lines.extend(segmented_lines)
            lines_count_dict[i] = len(segmented_lines)
    
    box_text, confidence = predict(lines)
    current_text_index = 0
    for i in range(len(bounding_box)):
        if i in exclude_index or i in null_image_index:
            continue
        
        # images are in reverse order
        text = ''.join(f'<p>{text}</p>' for text in reversed(box_text[current_text_index:current_text_index+lines_count_dict[i]]))
        text =  re.sub(r'\s+', ' ', text)
        extracted_text[i]=text
        extracted_text_confidence[i] = np.average(confidence[current_text_index:current_text_index+lines_count_dict[i]])
        current_text_index += lines_count_dict[i]

    return extracted_text, extracted_text_confidence

def extract_text_and_update_json(image, json_file):
    """
    Extracts text from an image and updates the provided JSON data with the extracted text.

    Parameters:
    - image (PIL.Image): The image from which to extract text.
    - json_file (dict): The JSON data to be updated. 
    Returns:
    - dict: The updated JSON data with the extracted text.
    """
    data = json_file

    bounding_boxes = data.get('jsonData', {}).get('boxes', [])
    classes = data.get('jsonData', {}).get('classes', [])
    
    indexes_of_excluded_class = [idx for idx, x in enumerate(classes) if x == "Table" or x == 'Picture']
    
    extracted_text, confidence = extract_text_by_boxes(image, bounding_boxes, indexes_of_excluded_class)
    

    data['jsonData']["extracted_text"] = extracted_text
    data['jsonData']["extracted_text_confidence"] = extracted_text

    return data
