import pickle
from google.cloud import storage
import os

def load_modelos(model_content, model_name):
    model = pickle.loads(model_content)
    return model

def load_models():
    # Load environment variables
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())

    # Authenticate and initialize client


    client = storage.Client(project=os.environ["PROJECT_ID"])

    # Load models
    model_names = ["VGG16_model_split2_total_personal_care", "model_accesories_vgg16", "model_footwear_vgg16", "model_master_vgg16"]
    loaded_models = {}
    for model_name in model_names:
        try:
            bucket = client.bucket(os.environ["BUCKET_NAME"])
            blob = bucket.blob(model_name)
            model_content = blob.download_as_string()
            loaded_model = load_modelos(model_content, model_name)
            loaded_models[model_name] = loaded_model
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")

    return loaded_models
