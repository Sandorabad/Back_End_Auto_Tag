from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from automatic_tagging.funciones.pipelines import preproc_pipeline
from PIL import Image
from io import BytesIO
import os
from automatic_tagging.funciones.modelos import load_model, pred
from automatic_tagging.funciones.gc_loader import load_models
from automatic_tagging.funciones.more_loaders import model_loaders
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.state.models = model_loaders()

@app.get('/')
def root():
    return {'greeting': 'Hello'}


@app.post("/pred/")

def prediction(file : UploadFile):

    image_bytes = file.file.read()

    preproc_image = preproc_pipeline(image_bytes)
    model_master = app.state.models["model_master_robust"]
    master_pred = pred(model_master,preproc_image,category = 'Master')



    if master_pred == 'Accessories':
        model= app.state.models["model_accesories_robust"]
        sub_pred = pred(model,preproc_image,category = 'Accessories')
    elif master_pred == 'Apparel':
        model = app.state.models["model_apparel_robust"]
        sub_pred = pred(model,preproc_image,category = 'Apparel')
    elif master_pred == 'Footwear':
        model = app.state.models["model_footwear_robust"]
        sub_pred = pred(model,preproc_image,category = 'Footwear')
    elif master_pred == 'Personal Care':
        model = app.state.models["model_personal_care_robust"]
        sub_pred = pred(model,preproc_image,category = 'Personal Care')
    else:
        print('Something went wrong')

    return {'master':master_pred, 'sub':sub_pred}
