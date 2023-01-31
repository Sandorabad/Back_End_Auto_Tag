from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from automatic_tagging.funciones.pipelines import final_pipeline
from PIL import Image
from io import BytesIO


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)



@app.get('/')
def root():
    return {'greeting': 'Hello'}


@app.post("/pred/")

def prediction(file : UploadFile):

    image_bytes = file.file.read()
    # byte_io = BytesIO()
    # image_bytes.save(byte_io, "JPEG")
    # image_bytes = byte_io.getvalue()

    result = final_pipeline(image_bytes)


    return result
