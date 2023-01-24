from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from automatic_tagging.funciones.pipelines import final_pipeline

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


@app.get("/pred")
def prediction():

    result = final_pipeline()
    return result
