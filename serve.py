
from fastapi import FastAPI
from DocumentStructureModel import DocumentStructureModel
from typing import List
from pydantic import BaseModel



app = FastAPI()



class ImageInput(BaseModel):
    base64_encoded_images: List[str]

@app.post("/api/v1.0/predictions")
async def root(X: ImageInput):    
    model = DocumentStructureModel()
    result = model.predict(X)
    
    return result