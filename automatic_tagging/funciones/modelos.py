# from keras.models import load_model
import tensorflow as tf
from automatic_tagging.funciones.f_tagger import tagger, cat_for_tag
# from automatic_tagging.funciones.procesamiento

pc_path="/home/vvdiaz1/code/vvdiaz1/automatic_tagging/model_weights/model_personalcare_vgg16"
f_path="/home/vvdiaz1/code/vvdiaz1/automatic_tagging/model_weights/model_footwear_vgg16"
ap_path="/home/vvdiaz1/code/vvdiaz1/automatic_tagging/model_weights/model_apparel_vgg16"
ac_path="/home/vvdiaz1/code/vvdiaz1/automatic_tagging/model_weights/model_accesories_vgg16"
m_path="/home/vvdiaz1/code/vvdiaz1/automatic_tagging/model_weights/model_master_vgg16"


def load_model(weights_path):
    model = tf.keras.models.load_model(weights_path)

    return model

def pred(model,preproc_image,category = 'Master'):
    # dict_categories = cat_for_tag()
    # if category == 'Master':
    #     categories = dict_categories['Master']
    # elif category == 'Accesories':
    #     categories = dict_categories['Accesories']
    # elif category == 'Apparel':
    #     categories = dict_categories['Apparel']
    # elif category == 'Footwear':
    #     categories = dict_categories['Footwear']
    # elif category == 'Personal Care':
    #     categories = dict_categories['Personal Care']
    # else:
    #     print('Something went wrong')

    predictions = model.predict(preproc_image)
    y_pred = tagger(predictions, category)

    return y_pred
