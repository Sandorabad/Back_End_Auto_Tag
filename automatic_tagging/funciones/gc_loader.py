
def load_models():
    from google.cloud import storage
    from google.auth.transport.requests import AuthorizedSession
    import os

    # Load environment variables
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())

    # Authenticate and initialize client
    import os
    import google.auth
    from google.oauth2.service_account import Credentials

    credentials = Credentials.from_service_account_file(os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
    authed_session = AuthorizedSession(credentials)
    client = storage.Client(project=os.environ["PROJECT_ID"], credentials=credentials)

    # Load models
    model_names = ["VGG16_model_split2_total_personal_care", "model_accesories_vgg16", "model_footwear_vgg16", "model_master_vgg16"]
    from automatic_tagging.funciones.modelos import load_model
    loaded_models = {}
    for model_name in model_names:
        bucket = client.bucket(os.environ["BUCKET_NAME"])
        blob = bucket.blob(model_name)
        model_content = blob.download_as_string()
        loaded_model = load_model(model_content, model_name)
        loaded_models[model_name] = loaded_model

    return loaded_models
