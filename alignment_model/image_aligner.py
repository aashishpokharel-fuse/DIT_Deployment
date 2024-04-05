import os 
import cv2
import sys
import time
import argparse
import json
import uvicorn
import numpy as np
from pathlib import Path
from collections import ChainMap
import torch


from alignment_model.orb_aligner import OrbAligner
from alignment_model.post_processing import homography_calculation
from alignment_model.post_processing import warp_image, normalization


def preprocess_image(input, IMAGE_SIZE = 1024):
    """
    Preprocesses an input image by resizing it to a specified size, normalizing its values,
    and converting it to the appropriate format for alignment model.

    Args:
        input (numpy.ndarray): The input image to preprocess.
        IMAGE_SIZE (int, optional): The desired size of the image after resizing. Defaults to 1024.

    Returns:
        torch.Tensor: The preprocessed image tensor.

    """
    input = cv2.resize(input, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    input = 1 - input/255
    # input = (input - 0.5)/1.0
    input = input.transpose(2, 0, 1)
    input = torch.from_numpy(input)
    input = input.unsqueeze(0)
    input = input.float()
    
    return input


def warp_image_to_reference(original_input_img, reference_img, homography_hat):
    """
        Warps the original input image to match the perspective of the reference image using a homography matrix.

        Parameters:
            original_input_img (ndarray): The original input image.
            reference_img (ndarray): The reference image.
            homography_hat (ndarray): The homography matrix.

        Returns:
            ndarray: The warped image.
    """

    data_image_2 = original_input_img.detach().cpu().numpy()
    rih,riw,_=reference_img.shape

    base_point = np.array([[0., 0., 1.], [0., 1024., 1.], [1024., 0., 1.], [1024., 1024., 1.]])

    # x & y
    ref_point = np.array([[0., 0., 1.], [0., float(rih), 1.], [float(riw), 0., 1.], [float(riw), float(rih), 1.]])

    image_2 = data_image_2
    homography_mat = homography_hat

    h_base_point = np.matmul(homography_mat, base_point.transpose())
    h_base_point = h_base_point/h_base_point[-1]
    h_base_point = h_base_point.transpose()
    h_base_point = np.delete(h_base_point, -1, axis=1)

    h_ref_point = []
    for i in h_base_point:
        f = [normalization(i[0], 1024, riw), normalization(i[1], 1024, rih)]
        h_ref_point.append(f)

    h_ref_point = np.array(h_ref_point)
    ref_homography_mat, _ = cv2.findHomography(ref_point, h_ref_point, cv2.RANSAC, 5.0)

    warp_image_2 = warp_image(data_image_2, ref_homography_mat, (rih, riw))

    warp_image_2 = warp_image_2.astype(np.uint8)

    warped_image = cv2.cvtColor(warp_image_2, cv2.COLOR_BGR2RGB)
    
    return warped_image


def post_process_aligned_img(img, ref_img=None):
    """
    Post-processes an aligned image using the reference image.

    Args:
        img: The image to be post-processed.
        ref_img: The reference image to align the input image to. (default: None)

    Returns:
        The post-processed image after alignment.

    """

    ho, wo,_ = ref_img.shape
    orb = OrbAligner()
    stat, h_matrix, num_matches, orb_img = orb.align_image(img, ref_img)

    if stat:
        print(f"ORB Stat: {stat}")
        warped_img = cv2.warpPerspective(img, h_matrix, dsize=(wo, ho), flags=cv2.INTER_LINEAR)

    return warped_img


def align_image(images, model, device):
    """
    Aligns an input image with a reference image using a given model and device.

    Args:
        images (List(numpy.ndarray)): The input image as a NumPy array.
        image_save_path (str, optional): The file path where the aligned image will be saved. If None, the image is not saved.
        model (torch.nn.Module): The model used for alignment.
        device (torch.device): The device used for alignment.

    Returns:
        numpy.ndarray: The aligned image as a NumPy array.
    """
    # original_reference_img = np.zeros_like(original_input_img)
    original_reference_imgs = images # used for homography matrix calculation for original image size

    img = np.ones((1024, 1024, 3)) * 255
    dx = 50
    dy = 50
    grid_color = [0, 0, 0]

    img[::dy, :, :] = grid_color
    img[:, ::dx, :] = grid_color
    img = img/255
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    reference_img = img.float()
    
    original_input_imgs = [torch.from_numpy(image).unsqueeze(0).float() for image in images]
    processed_input_imgs = [preprocess_image(images) for images in images]
    processed_reference_imgs = [reference_img for _ in range(len(images))]

    batch_size = 16
    
    with torch.no_grad():
        model.eval()
        aligned_images = []
        
        for i in range(0, len(processed_input_imgs), batch_size):
            input_images = processed_input_imgs[i:i+batch_size]
            original_images = original_input_imgs[i:i+batch_size]
            ref_images = processed_reference_imgs[i:i+batch_size]
            original_ref_images = original_reference_imgs[i:i+batch_size]
            
            inputs = torch.cat(input_images, dim=0).to(device)
            ref_imgs = torch.cat(ref_images, dim=0).to(device)
            
            outputs = model(inputs, ref_img=ref_imgs)
            outputs = outputs.detach().cpu()
            for j in range(len(outputs)):
                homography_hat = homography_calculation(outputs[j].numpy(), input_images[j]) 
                aligned_image = warp_image_to_reference(original_images[j].squeeze(0), original_ref_images[j], homography_hat)
                
                # disabled since we are not using reference image
                # aligned_image = post_process_aligned_img(original_input_img, original_reference_img)

                aligned_images.append(aligned_image)
        
        return aligned_images