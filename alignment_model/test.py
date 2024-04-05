import unittest

import fitz
import numpy as np
import torch
import yaml
import os

from pdf2image import convert_from_path

from alignment_model.model import DVQAModel
from inference import alignment_model_inference

def load_config(config_name):
    with open(os.path.join('', config_name)) as file:
        config = yaml.safe_load(file)
    return config


config = load_config("/home/fm-pc-lt-173/Projects/extraction_engine_demo_ml/config.yaml")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(device)
alignment_model = DVQAModel().to(device)
checkpoint = torch.load(config['alignment_model_path'], map_location=device)
alignment_model.load_state_dict(checkpoint['model'])
alignment_model.to(device)

class TestAlignmentModelInference(unittest.TestCase):
    def setUp(self):
        self.image_dir = "/home2/Financial_Statement/COR/images"
        self.images = [os.path.join(self.image_dir, img) for img in os.listdir(self.image_dir) if img.endswith(".jpg")]
        self.save_dir = "/home2/Financial_Statement/COR/aligned_images"
    def test_process_image_true(self):
        aligned_images = alignment_model_inference(alignment_model, device, self.images, self.save_dir, True)
        self.assertEqual(len(aligned_images), len(self.images))
        for image in aligned_images:
            self.assertIsInstance(image, np.ndarray)

    def test_process_image_false(self):
        aligned_images = alignment_model_inference(alignment_model, device, self.images, self.save_dir, False)
        self.assertEqual(len(aligned_images), len(self.images))
        for image in aligned_images:
            self.assertIsInstance(image, np.ndarray)

    def test_save_dir(self):
        alignment_model_inference(alignment_model, device, self.images, self.save_dir, True)
        self.assertTrue(os.path.exists(self.save_dir))
        self.assertEqual(len(os.listdir(self.save_dir)), len(self.images))

        # Check if images are saved correctly
        for i in range(len(self.images)):
            image_path = os.path.join(self.save_dir, f"{i}.jpg")
            self.assertTrue(os.path.exists(image_path))

    def test_exception_handling(self):
        with self.assertRaises(Exception):
            alignment_model_inference(alignment_model, device, "invalid_input", self.save_dir, True)


if __name__ == "__main__":
        unittest.main()