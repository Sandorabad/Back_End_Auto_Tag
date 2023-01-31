# import procesamiento
from automatic_tagging.funciones.modelos import load_model, pred

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
import numpy as np
import os
from PIL import Image
from io import BytesIO

pc_path="/root/code/vvdiaz1/automatic_tagging/automatic_tagging/modelos/VGG16_model_split2_total_personal_care"
f_path="/root/code/vvdiaz1/automatic_tagging/automatic_tagging/modelos/model_footwear_vgg16"
#ap_path="/home/federico/code/vvdiaz1/automatic_tagging/model_weights/model_apparel_vgg16"
ac_path="/root/code/vvdiaz1/automatic_tagging/automatic_tagging/modelos/model_accesories_vgg16"
m_path="/root/code/vvdiaz1/automatic_tagging/automatic_tagging/modelos/model_master_vgg16"


#esta va en preprocesamiento.py
def img_byte_to_tensor(image_bytes):
    "recibe la imagen ya resized en byteio y la convierte a tensor (listo para meter a un modelo)"

   # Open image from bytes
    img = Image.open(image_bytes)

    # convert image to tensor and expand dims if neccesary
    #
    img_tensor = tf.expand_dims(img, axis = 0 )


    return img_tensor

#este tmbn va en preprocesamiento.py
def resize_img_byte(image_bytes):
    """recibe la imagen en bytes io y hace el resize, retorna un byte io"""
    width = 360
    height = 480
   # open image from bytes
    img = Image.open(BytesIO(image_bytes))

    # resize image
    img = img.resize((width, height), Image.ANTIALIAS)


    # save image back to bytes
    img_bytes = BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    print('img resized correctly')
    return img_bytes

def preproc_pipeline(image_bytes):
    #recibe este byteIO
    #hace el resize y tira byte io
    #pasar byte io a array para q entre al modelo

    image = resize_img_byte(image_bytes)
    img_tensor = img_byte_to_tensor(image)

    return img_tensor


def final_pipeline(image_bytes):
    """Recibe la imagen, la preprocesa, llama al camino de modelos correcto
    y luego retorna un diccionario con las categorias predichas"""

    preproc_image = preproc_pipeline(image_bytes) #sale como tensor
    model_master = load_model(m_path)
    master_pred = pred(model_master,preproc_image,category = 'Master')



    if master_pred == 'Accessories':
        model= load_model(ac_path)
        sub_pred = pred(model,preproc_image,category = 'Accessories')
    #elif master_pred == 'Apparel':
        #model = load_model(ap_path)
        #sub_pred = pred(model,preproc_image,category = 'Apparel')
    elif master_pred == 'Footwear':
        model = load_model(f_path)
        sub_pred = pred(model,preproc_image,category = 'Footwear')
    elif master_pred == 'Personal Care':
        model = load_model(pc_path)
        sub_pred = pred(model,preproc_image,category = 'Personal Care')
    else:
        print('Something went wrong')

    return {'Master Category':master_pred, 'Sub Category':sub_pred}


# import io
# from PIL import Image

<<<<<<< HEAD
#with open(os.getenv("TEST_IMG_PATH"), 'rb') as f:
#    image_bytes = f.read()

# resize_img_byte(image_bytes)
#x = final_pipeline(image_bytes)
#print(x)
=======
# img_path_acc = '/home/federico/code/vvdiaz1/automatic_tagging/raw_data/split1_total/test/Accessories/1599.jpg'
# img_path_ap = "/home/federico/code/vvdiaz1/automatic_tagging/raw_data/split1_total/test/Apparel/1533.jpg"
# img_path_pc  = "/home/federico/code/vvdiaz1/automatic_tagging/converse4.jpg"

# with open(img_path_ap, 'rb') as f:
#     image_bytes = f.read()

# # resize_img_byte(image_bytes)
# x = final_pipeline(image_bytes)
# print(x)
>>>>>>> e3ea46019acb054c2889c518b563c585d4ed2379
