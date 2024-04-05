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
import pytesseract
import re

def image_resolution_from_json(json_data):
    """
    Extracts the image resolution from base64-encoded image data in JSON.

    Parameters:
    - json_data (dict): JSON data containing the 'image' field.

    Returns:
    - tuple: Image resolution (width, height).
    """
    image_data = base64.b64decode(json_data['jsonData'].get('image', b''))
    # image = Image.open(BytesIO(image_data))
    image = Image.open(BytesIO(image_data))
    return image.size


def pdf_resolution(pdf_document):
    """
    Retrieves the width and height of the first page in a PDF document.

    Parameters:
    - pdf_file (str): Path to the PDF file.

    Returns:
    - tuple: PDF page width and height.
    """
    # pdf_document = fitz.open(pdf_file)
    # pdf_bytes = open(pdf_path, 'rb')
    # The line `pdf_document = fitz.open(stream=pdf_bytes, filetype='pdf')` is opening the PDF
    # document using the `fitz` library. It takes the `pdf_bytes` variable, which contains the binary
    # data of the PDF file, and opens it as a PDF document. The `filetype='pdf'` argument specifies
    # that the file is in PDF format.
    # pdf_document =  fitz.open(stream=pdf_bytes, filetype='pdf')
    page = pdf_document[0]
    image = page.get_pixmap()
    return image.width, image.height


def adjust_coordinates(bbox, scale_x, scale_y):
    """
    Adjusts bounding box coordinates based on scaling factors.

    Parameters:
    - bbox (tuple): Bounding box coordinates (left, top, right, bottom).
    - scale_x (float): Scaling factor for the x-axis.
    - scale_y (float): Scaling factor for the y-axis.

    Returns:
    - tuple: Adjusted bounding box coordinates.
    """
    return (
        bbox[0] * scale_x,
        bbox[1] * scale_y,
        bbox[2] * scale_x,
        bbox[3] * scale_y
    )

def extract_lines_from_text_image(image, img_index):
    """
    Extracts lines from the given image.

    Args:
        image: The text image from which to extract lines.

    Returns:
        A list of extracted lines (regions of interest) from the image.
    """
    gray_image = image.convert('L')
    blur = cv2.GaussianBlur(np.array(gray_image.copy()), (3, 3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
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
            if h >= 10:
                lines.append(image.crop((x-5,y-5,x+w+5,y+h+5)))
            
    else:
        lines.append(image)
    
    return lines

def extract_text_by_boxes(args, tsr_parser=False):
    """
    Extracts text from a PDF page within specified bounding boxes.

    Parameters:
    - args (tuple): Tuple containing PIL page image, adjusted bounding boxes, and index of label to exclude i.e. table.

    Returns:
    - list: List of extracted text from each bounding box.
    """
    page, adjusted_bounding_boxes, index = args
    extracted_text = []
    for i, box in enumerate(adjusted_bounding_boxes):
               
        if i in index:
            extracted_text.append("")

        else:
            left, top, right, bottom = box
            # rect = fitz.Rect(left, top, right, bottom)
            rect = page.crop((left,top,right,bottom))
            lines = extract_lines_from_text_image(rect,i)
            
            box_text = []  # Accumulate text items within a bounding box as a single string
            for line in lines:
                box_text.append(pytesseract.image_to_string(line))
            
            box_text = '\n'.join(box_text[::-1])
            box_text =  re.sub(r'\s+', ' ', box_text)
            
            extracted_text.append(box_text)

    return extracted_text


def pdf_processor(image, adjusted_bounding_boxes, index, target_page):
    """
    Process PDF pages in parallel to extract text from specified bounding boxes.

    Parameters:
    - pdf_document: PDF document opened with fitz.
    - adjusted_bounding_boxes (list): List of adjusted bounding boxes.
    - index (int): Index of the bounding box to exclude.

    Returns:
    - list: List of extracted text from all pages.
    """
    args_list = [(image, adjusted_bounding_boxes, index)]
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(extract_text_by_boxes, args_list))
    return [text for result in results for text in result]


# def process_pdf_and_update_json(pdf_path, json_path):
#     """
#     Main function to process PDF and update JSON data with extracted text.

#     Parameters:
#     - pdf_path (str): Path to the PDF file.
#     - json_path (str): Path to the JSON file.
#     """
#     with open(json_path, 'r', encoding='utf-8') as json_file, fitz.open(pdf_path) as pdf_document:
#         data = json.load(json_file)
#         bounding_boxes = data.get('jsonData', {}).get('boxes', [])
#         classes = data.get('jsonData', {}).get('classes', [])
#         index_of_excluded_class = classes.index(8)

#         # Calculate scaling factors based on image and PDF resolutions
#         image_size = image_resolution_from_json(data)
#         pdf_size = pdf_resolution(pdf_path)
#         scale_x = pdf_size[0] / image_size[0]
#         scale_y = pdf_size[1] / image_size[1]

#         # Adjust bounding boxes
#         adjusted_bounding_boxes = [adjust_coordinates(bbox, scale_x, scale_y) for bbox in bounding_boxes]

#         # Process pages in parallel
#         extracted_text = pdf_processor(pdf_document, adjusted_bounding_boxes, index_of_excluded_class)

#         # Update JSON data
#         data["extracted_text"] = extracted_text

#         with open(json_path, 'w', encoding='utf-8') as updated_json_file:
#             json.dump(data, updated_json_file, ensure_ascii=False, indent=2)

def process_pdf_and_update_json(image, json_file,target_page=0):
    """
    Main function to process PDF and update JSON data with extracted text.

    Parameters:
    - pdf_path (str): Path to the PDF file.
    - json_path (str): Path to the JSON file.
    """
    # with open(json_path, 'r', encoding='utf-8') as json_file, fitz.open(pdf_path) as pdf_document:
    # data = json.load(json_file)
    data = json_file
    # pdf bytes
    # pdf_document = fitz.open(stream=pdf_file, filetype='pdf')
    bounding_boxes = data.get('jsonData', {}).get('boxes', [])
    classes = data.get('jsonData', {}).get('classes', [])
    # print("BBOX CLASSES",bounding_boxes, classes)
    # try:
    #     index_of_excluded_class = classes.index(8)
    # except ValueError:
    #     index_of_excluded_class = classes.index("Table") # This gets a single table Needs to be handled later
    indexes_of_excluded_class = [idx for idx, x in enumerate(classes) if x == "Table" or x == 'Picture']
    # print("PRINT", indexes_of_excluded_class)
    # Calculate scaling factors based on image and PDF resolutions
    image_size = image_resolution_from_json(data)
    # pdf_size = pdf_resolution(pdf_document)
    # scale_x = pdf_size[0] / image_size[0]
    # scale_y = pdf_size[1] / image_size[1]

    # Adjust bounding boxes
    # adjusted_bounding_boxes = [adjust_coordinates(bbox, scale_x, scale_y) for bbox in bounding_boxes]

    # Process pages in parallel
    extracted_text = pdf_processor(image, bounding_boxes, indexes_of_excluded_class, target_page)

    # Update JSON data
    data['jsonData']["extracted_text"] = extracted_text

    # with open(json_path, 'w', encoding='utf-8') as updated_json_file:
    #     json.dump(data, updated_json_file, ensure_ascii=False, indent=2)

    return data


# Measure the processing time
start_time = time.time()

if __name__ == "__main__":
    # Paths to the PDF and JSON files
    pdf_path = "F:\FUSE\EXTRACTION ENGINE\main-entry-point\Invoice.pdf"
    json_path = "F:\FUSE\EXTRACTION ENGINE\main-entry-point\Invoice.json"

    with open(pdf_path, 'rb') as pdf_document, open(json_path, 'r', encoding='utf-8') as json_bytes:
        jsons = json.load(json_bytes)
        pdf_bytes = pdf_document.read()
        data = process_pdf_and_update_json(pdf_bytes, jsons)

        # print(data)

# Print the processing time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Processing time: {elapsed_time} seconds")
