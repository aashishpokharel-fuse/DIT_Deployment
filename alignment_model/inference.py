import os
import cv2
import sys
import time
import argparse
import json
import torch
import uvicorn
import numpy as np
from pathlib import Path
from collections import ChainMap
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, conint
from typing import Optional, List, Union
import numpy as np
from io import BytesIO
from PIL import Image

from alignment_model.image_aligner import align_image
# from model import DVQAModel

app = FastAPI()

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load model
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# # print(device)
# model = DVQAModel().to(device)
#
# # model_save_path = ('/home/ubuntu/Info-Extraction-Engine-ML/modified_code/extraction_engine_demo_ml/'
# #                    'alignment_model/model/checkpoints/dvqa_model.pth')
#
# model_save_path = 'alignment_model/model/checkpoints/dvqa_model.pth'
#
# checkpoint = torch.load(model_save_path, map_location=device)
# model.load_state_dict(checkpoint['model'])
# model.to(device)
# print("Alignment model loaded")


class AlignInput(BaseModel):
    """
    A Pydantic model that represents the input for the image alignment API.
    It contains optional fields for image paths and a save directory.

    Attributes:
        image_paths (List[str]): A list of paths to the images to be processed.
        save_dir (Optional[str]): The directory where the processed images will be saved.
    """
    image_paths: List[str] = []
    save_dir: Optional[str] = None


@app.post("/align")
async def run_inference(model, device, item: AlignInput):
    """
    It accepts an instance of ROIExtractionInput which can contain a list of image paths,
    a save directory, and a list of numpy images as input.
    Each image is processed and aligned using a pre-defined model and device.
    The aligned images are saved in the provided directory (if any) and returned in the response.

    Args:
        model: loaded model
        device: str, device type to run model on
        item (ROIExtractionInput): An instance of ROIExtractionInput containing the images to be processed and the save directory.

    Raises:
        HTTPException: If there is an error processing the images.

    Returns:
        dict: A message indicating the success of the operation and the aligned images.
    """
    try:
        image_paths = item.image_paths
        save_dir = item.save_dir

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        aligned_images = []

        for index, image in enumerate(image_paths):
            image = Image.open(image)

            if save_dir:
                aligned_image_save_dir = os.path.join(save_dir, f"{index}.jpg")
            else:
                aligned_image_save_dir = None

            aligned_image = align_image(image, aligned_image_save_dir, model, device)
            aligned_images.append(aligned_image.tolist())

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing: {str(e)}")

        return {"message": "ROI extraction completed successfully",
                "aligned_images": json.dumps(aligned_images)}


@app.post("/align/upload")
async def run_inference_upload(model, device, files: List[UploadFile] = File(default=None),
                               save_dir: Optional[str] = Form(default=None)):
    """
    This function handles POST requests to the "/align/upload" endpoint.
    It accepts a list of files and an optional save directory as input.
    Each file is processed and aligned using a alignment model.
    The aligned images are saved in the provided directory (if any) and returned in the response.

    Args:
        model: loaded model
        device: str, device to run model on
        files (List[UploadFile], optional): List of image files to be processed. Defaults to None.
        save_dir (str, optional): Directory where the processed images will be saved. Defaults to None.

    Raises:
        HTTPException: If there is an error processing the images.

    Returns:
        dict: A message indicating the success of the operation and the aligned images.
    """
    try:
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        aligned_images = []
        for index, image in enumerate(files):
            image = Image.open(BytesIO(image.file.read()))

            if save_dir:
                aligned_image_save_dir = os.path.join(save_dir, f"{index}.jpg")
            else:
                aligned_image_save_dir = None

            aligned_image = align_image(image, aligned_image_save_dir, model, device)
            aligned_images.append(aligned_image.tolist())

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing: {str(e)}")

    return {"message": "ROI extraction completed successfully",
            "aligned_images": json.dumps(aligned_images)}


def alignment_model_inference(model, device, images):
    """
    Accepts list of images, and returns list aligned image

    Args:
        model: loaded model
        device: str, device to run model on
        images (List[Union[np.ndarray|PIL.Image, str]]): The list of images to be aligned. Each image can be a numpy array or a string representing the image path.
        process_image (bool, optional): If True, the function will process and align the images. If False, the function will return the original images. Defaults to True.

    Returns:
        List[np.ndarray]: The list of aligned images.
    """
    try:
        input_images = []

        for index, image in enumerate(images):
            if isinstance(image, str):
                input_image = cv2.imread(image)
                input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            else:
                input_image = np.array(image)
            input_images.append(input_image)
        aligned_images = align_image(input_images, model, device)

    except Exception as e:
        print(e)
        raise Exception(f"Error processing: {str(e)}")

    return aligned_images

def correct_orientation(model, device, img):
    """
    This function accepts an image and corrects its orientation using a Paddle PUCL model.

    Args:
        model: loaded model
        device: str, device to run model on
        img (np.ndarray): The input image as a NumPy array.

    Returns:
        np.ndarray: The corrected image as a NumPy array.
    """
    try:
        corrected_image = img
        result = model.predict(input_data=corrected_image)
        result = next(result)[0]

        initial_label = result['label_names'][0]

        if initial_label == "180":
            corrected_image = cv2.rotate(corrected_image, cv2.ROTATE_180)
            result = model.predict(input_data=corrected_image)
            result = next(result)[0]
            initial_label = result['label_names'][0]

            if initial_label != "0":
                corrected_image = cv2.rotate(corrected_image, cv2.ROTATE_180)

        original_out_img = corrected_image.copy()

        if initial_label != "0":
            if initial_label in ["90", "270"]:
                corrected_image = cv2.rotate(original_out_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
                result = model.predict(input_data=corrected_image)
                result = next(result)[0]

                initial_label = result['label_names'][0]

                if initial_label != "0":
                    corrected_image = cv2.rotate(original_out_img, cv2.ROTATE_90_CLOCKWISE)
                    
                    result = model.predict(input_data=corrected_image)
                    result = next(result)[0]

                    initial_label = result['label_names'][0]
                    initial_confidence = result['scores'][0]

        if initial_label != "0":
            result = model.predict(input_data=original_out_img)
            result = next(result)[0]
            initial_label = result['label_names'][1]
            
            if initial_label == "0":
                corrected_image = original_out_img

    except Exception as e:
        raise Exception(f"Error processing: {str(e)}")

    return corrected_image


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)