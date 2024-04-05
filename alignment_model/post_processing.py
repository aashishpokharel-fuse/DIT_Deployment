import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

def destandardize(image, mean=0.5, std=1.0, transpose=False):
    image = ((image * std) + mean) * 255.
    image = np.rint(image)
    image[image > 255] = 255
    image[image < 0] = 0

    if transpose:
        image = image.transpose((1, 2, 0))

    return image


def warp_image(image, homography, target_hw):
    h, w = target_hw
    homography = np.linalg.inv(homography)
    return cv2.warpPerspective(image, homography, dsize=(w, h))


def normalization(x, max_val, new_max_val):
    return (x/max_val) * new_max_val

def homography_calculation(perspective_field, input_img):
    perspective_field = perspective_field.reshape(2, -1).T
    h, w = input_img.shape[2:4]
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    point_grid = np.vstack((x_grid.flatten(), y_grid.flatten())).transpose()

    predicted_point_gird = point_grid + perspective_field

    homography, _ =  cv2.findHomography(point_grid, predicted_point_gird, cv2.RANSAC, 5.0)

    # homography_hats = torch.tensor(homography, dtype=torch.float32, device=input_img.device)
    homography_hats = homography.astype(np.float32)
    return homography_hats