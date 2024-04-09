import io
import torch

import numpy as np 
from PIL import Image
import time

# OCR and Alignment Model packages Imports
from alignment_model.model import DVQAModel
from alignment_model.inference import alignment_model_inference, correct_orientation
from paddleclas import PaddleClas


from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from batch_processing import BatchPredictor
import torch
import base64

from ditod import add_vit_config
from src.config import load_model, CONFIG_PATH, CONFIG
import os
from postprocess import postprocess_dit


class DocumentStructureModel(object):

    def __init__(self):
        self.loaded = False
        load_model()
    
    def load(self):
        
        
        
        # MODELNAME = os.environ.get('MODELNAME')
        # VERSION = os.environ.get('VERSION').replace(".", "-")
        # MODEL_FILE_NAME = os.environ.get('MODEL_FILE_NAME')
        # MODEL_PATH = f"./deployment/{MODELNAME}/{VERSION}-{MODEL_FILE_NAME}"
        # print("CALLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL")
        print("**********************************************************")
        print("******************LOADING CONFIG**************************")
        print("**********************************************************")
        
        print(CONFIG)
        MODEL_PATH = './model/dit/model_weights.pth'
        MODEL = ["MODEL.WEIGHTS", MODEL_PATH]

        self.cfg = get_cfg()
        add_vit_config(self.cfg)
        self.cfg.merge_from_file(CONFIG_PATH)
        self.cfg.merge_from_list(MODEL)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("DEVICE:",self.device)
        # device = 'cuda'
        # DIT MODEL
        self.cfg.MODEL.DEVICE = self.device
        self.predictor = BatchPredictor(self.cfg)
        
        # Init OCR & ALIGNMENT MODEL
        self.alignment_model = DVQAModel().to(self.device)
        checkpoint = torch.load(CONFIG['alignment_model_path'], map_location=self.device)
        self.alignment_model.load_state_dict(checkpoint['model'])
        self.alignment_model.to(self.device)
        self.orientation_model = PaddleClas(
                               inference_model_dir=CONFIG['pulc_model_path'], 
                               use_gpu=False, 
                               class_id_map_file=CONFIG['pulc_class_id_map_file'],
                               resize_short=1120,
                               crop_size=1120,
                               )
        
        self.loaded = True
        print("Loaded model")

    def predict(self, X, feature_names = None):
        print("X IS:",type(X))
        if isinstance(X, list) or isinstance(X, np.ndarray):
            img = []
            for item in X:
                decoded =   base64.b64decode(item)
                decoded_image = np.array(Image.open(io.BytesIO(decoded)).convert("RGB"))[:, :, ::-1]
                img.append(decoded_image)
        else:
            decoded =   base64.b64decode(X)
            decoded_image = np.array(Image.open(io.BytesIO(decoded)).convert("RGB"))[:, :, ::-1]
            img     =   [decoded_image ,]
        
        md      =   MetadataCatalog.get(self.cfg.DATASETS.TEST[0])
        if self.cfg.DATASETS.TEST[0]=='icdar2019_test':
            md.set(thing_classes=["table"])
        else:
            md.set(thing_classes=['Caption', 'Footnote', 'Formula', 'List-item', 'Pagefooter', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text','Title', 'Handwriting', 'Stamps'])
        with torch.no_grad():
            # outputs = self.predictor(img)["instances"]
            
            # #################### INFERENCE ############################# #
            imgs, encoded_images = self.get_aligned(img)
            outputs = self.predictor(imgs)
            ################################################################
            
            # # Encoding the Aligned Images
            # encoded_images= []
            # for aligned_image in img:
            #     image_pil = Image.fromarray(aligned_image)        
            #     image_bytes = io.BytesIO()
            #     image_pil.save(image_bytes, format='png')
            #     encoded_images.append(str(base64.b64encode(image_bytes.getvalue()))[2:-1])
                
        
            # print("The OUTPUT Length:",len(outputs))
        labels = ['Caption', 'Footnote', 'Formula', 'List-item', 'Pagefooter', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text','Title', 'Handwriting', 'Stamps']
        final_op = []
        for idx,output in enumerate(outputs):
            output= output['instances']
            boxes   = [item.tolist() for item in output.get_fields()['pred_boxes']]
            scores  = [item.tolist() for item in output.get_fields()['scores']]
            classes = [item.tolist() for item in output.get_fields()['pred_classes']]
            classes = [labels[item]  for item in classes]
            
            output_json  =   {   
                            "scores": list(scores), 
                            "classes":list(classes),
                            "boxes": list(boxes), 
                            "image":encoded_images[idx],
                        }
            
            final_op.append(output_json)
            
        batch_results = {"results": final_op}
        # This section is currently done to match the output format for the current demo 
        # (Will be removed in the next refactoring)
        batch_results = self.format_output(batch_results)
        batch_results_ocr = []
        for batch_result in batch_results:
            batch_result_post = postprocess_dit(batch_result)
            batch_result_ocr = self.extract_text(batch_result_post)
            batch_results_ocr.append(batch_result_ocr)
        return {"results":batch_results_ocr}
    
    def get_aligned(self, images):
        start_time = time.time()
        corrected_images = [correct_orientation(self.orientation_model, self.device, np.array(img)) for img in images]
        print("Orientation model time", time.time() - start_time)
        start_time = time.time()
        
        aligned_images = alignment_model_inference(self.alignment_model, self.device,
                                                corrected_images)
        print("Alignment model time", time.time() - start_time)
        
        # This code snippet is iterating over a list of aligned images and performing the following
        # operations for each image:
        encoded_images = []
        for aligned_image in aligned_images:
            image_pil = Image.fromarray(aligned_image)        
            image_bytes = io.BytesIO()
            image_pil.save(image_bytes, format='png')
            encoded_images.append(str(base64.b64encode(image_bytes.getvalue()))[2:-1])
        return aligned_images, encoded_images
    
    
    def format_output(self, output):
        # start_time = time.time()
        response_list = []
        try:
            response_list = [{"jsonData":item} for item in output['results']]
        except Exception:
            print("EXCEPTION",jsondata)
            
        # print("DIT model Time : ", time.time() - start_time)
        return response_list
        
    def extract_text(self, dit_output):
        from ocr.extract_text_custom_ocr import extract_text_and_update_json

        start_time = time.time()
        decoded = base64.b64decode(dit_output['jsonData']['image'])
        aligned_image = Image.open(io.BytesIO(decoded)).convert("RGB")
        # aligned_image.save(f"/mnt/gime-extract/global_ime/data/output/aligned_image/{pdf.split('/')[-1]}")
        text_extract = extract_text_and_update_json(aligned_image, dit_output)
        print("Extract Text Time : ", time.time() - start_time)
        
        return text_extract
        