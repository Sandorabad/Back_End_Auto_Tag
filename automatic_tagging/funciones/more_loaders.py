import requests
import os
import tensorflow as tf
def model_loaders():
    models_urls = [
    "https://storage.googleapis.com/modelos_auto_tag/model_personal_care_robust",
    "https://storage.googleapis.com/modelos_auto_tag/model_master_robust",
    "https://storage.googleapis.com/modelos_auto_tag/model_footwear_robust",
    "https://storage.googleapis.com/modelos_auto_tag/model_apparel_robust",
    "https://storage.googleapis.com/modelos_auto_tag/model_accesories_robust"
]

    models = {}

    for url in models_urls:
        filename = url.split("/")[-1] + ".h5"
        response = requests.get(url)
        open(filename, "wb").write(response.content)
        model = tf.keras.models.load_model(filename)
        models[filename.split(".")[0]] = model

    return models
