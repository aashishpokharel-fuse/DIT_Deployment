import io
import torch

import numpy as np 
from PIL import Image

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from batch_processing import BatchPredictor
import torch
import base64

from ditod import add_vit_config
from src.config import load_model, CONFIG_PATH
import os


class DocumentStructureModel(object):

    def __init__(self):
        self.loaded = False
        load_model()
    
    def load(self):
        
        
        
        # MODELNAME = os.environ.get('MODELNAME')
        # VERSION = os.environ.get('VERSION').replace(".", "-")
        # MODEL_FILE_NAME = os.environ.get('MODEL_FILE_NAME')
        # MODEL_PATH = f"./deployment/{MODELNAME}/{VERSION}-{MODEL_FILE_NAME}"
        
        MODEL_PATH = './model/model_weights.pth'
        MODEL = ["MODEL.WEIGHTS", MODEL_PATH]

        self.cfg = get_cfg()
        add_vit_config(self.cfg)
        self.cfg.merge_from_file(CONFIG_PATH)
        self.cfg.merge_from_list(MODEL)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg.MODEL.DEVICE = device
        self.predictor = BatchPredictor(self.cfg)
        self.loaded = True
        print("Loaded model")

    def predict(self, X, feature_names = None):
        # img = np.array(Image.open(io.BytesIO(X)).convert("RGB"))[:, :, ::-1]
        # encoded = base64.b64encode(X.read())
        print("X IS:",type(X))
        # X = io.BytesIO(X)
        # print("X IS:",type(X))
        if isinstance(X, list) or isinstance(X, np.ndarray):
            img = []
            for item in X:
                decoded =   base64.b64decode(item)
                # print("DECODED", decoded)
                print("TYPE DECODED", type(decoded))
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
            outputs = self.predictor(img)
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
                            "image":X[idx],
                        }
            
            final_op.append(output_json)
        batch_results = {"results": final_op}
        return batch_results