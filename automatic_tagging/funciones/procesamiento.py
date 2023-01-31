
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.utils import to_categorical

import os
from skimage.io import imread
import matplotlib.pyplot as plt
from PIL import Image


def resize_images(original_folder = os.getenv("LOCAL_IMAGES_PATH"),
                  save_folder = os.getenv("RESIZED_REGISTRY_PATH"),
                  width = 360, height = 480):

#original_folder and save_folder are folder paths located in the .env script
#save_folder must exist before executing, create using mkdir in terminal
    for filename in os.listdir(original_folder):
    # Open the image file
        img = Image.open(os.path.join(original_folder, filename))

    # Resize the image
        img = img.resize((width, height))
    #Rescale image pixels
        img_array = np.array(img)
        img_array = img_array / 255.0

        img_rescaled = Image.fromarray(img_array)

    # Save the resized image in the resized folder
        img_rescaled.save(os.path.join(save_folder, filename))
    print(f"Images rescaled, resized and saved to {save_folder}, Ready for model")
    return None


def target_creation(styles_folder = os.getenv("STYLES_CSV_PATH")):

    #Se carga el DF del Target
    df_styles = pd.read_csv(styles_folder, error_bad_lines=False)

    #Labels 1 corresponde a la master category
    labels1 = np.unique(df_styles.masterCategory).tolist()
    num_cat1 = len(labels1)

    #Se define el masterCaterogory y se transforma a variable to_categorical
    y1 = df_styles[['masterCategory']]
    y1.masterCategory =pd.Categorical(y1.masterCategory)
    y1_num = y1.masterCategory.cat.codes
    y1_cat = to_categorical(y1_num,num_cat1)

    #Labels para el segundo modelo/Segunda jerarquia de categorias
    labels2 = np.unique(df_styles[df_styles['masterCategory'] == labels1[0]].subCategory).tolist()
    num_cat2 = len(labels2)

    #definimos y como el subcategory que corresponde al masterCategory

    y2 = df_styles[df_styles['masterCategory'] == labels1[0]][['subCategory']]
    y2.subCategory = pd.Categorical(y2.subCategory)
    y2_cat = to_categorical(y2.subCategory.cat.codes,len(labels2))

    #3era jerarquia de categorias
    labels3 = np.unique(df_styles[df_styles['masterCategory'] == labels1[1]].subCategory).tolist()
    num_cat3 = len(labels3)

    #definimos y como el subcategory que corresponde al masterCategory

    y3 = df_styles[df_styles['masterCategory'] == labels1[1]][['subCategory']]
    y3.subCategory = pd.Categorical(y3.subCategory)
    y3_cat = to_categorical(y3.subCategory.cat.codes,len(labels3))

    #4ta jerarquia de categoria
    labels4 = np.unique(df_styles[df_styles['masterCategory'] == labels1[2]].subCategory).tolist()
    num_cat4 = len(labels4)


    #definimos y como el subcategory que corresponde al masterCategory
    y4= df_styles[df_styles['masterCategory'] == labels1[2]][['subCategory']]
    y4.subCategory = pd.Categorical(y4.subCategory)
    y4_cat = to_categorical(y4.subCategory.cat.codes,len(labels4))

    #5ta jerarquia
    labels5 = np.unique(df_styles[df_styles['masterCategory'] == labels1[3]].subCategory).tolist()
    num_cat5 = len(labels5)


    #definimos y como el subcategory que corresponde al masterCategory
    y5= df_styles[df_styles['masterCategory'] == labels1[3]][['subCategory']]
    y5.subCategory = pd.Categorical(y5.subCategory)
    y5_cat = to_categorical(y5.subCategory.cat.codes,len(labels5))

    #6to Jerarquia
    labels6 = np.unique(df_styles[df_styles['masterCategory'] == labels1[4]].subCategory).tolist()
    num_cat6 = len(labels6)

    y6= df_styles[df_styles['masterCategory'] == labels1[4]][['subCategory']]
    y6.subCategory = pd.Categorical(y6.subCategory)
    y6_cat = to_categorical(y6.subCategory.cat.codes,len(labels6))

    #7to Jerarquia
    labels7 = np.unique(df_styles[df_styles['masterCategory'] == labels1[5]].subCategory).tolist()
    num_cat7 = len(labels7)


    y7= df_styles[df_styles['masterCategory'] == labels1[5]][['subCategory']]
    y7.subCategory = pd.Categorical(y7.subCategory)
    y7_cat = to_categorical(y7.subCategory.cat.codes,len(labels7))

    #8va Jerarquia
    labels8 = np.unique(df_styles[df_styles['masterCategory'] == labels1[6]].subCategory).tolist()
    num_cat8 = len(labels8)

    y8= df_styles[df_styles['masterCategory'] == labels1[6]][['subCategory']]
    y8.subCategory = pd.Categorical(y8.subCategory)
    y8_cat = to_categorical(y8.subCategory.cat.codes,len(labels8))

    #Armar diccionariado de arrays
    full_df = {"Master" : y1_cat, "Accesories" : y2_cat, "Apparel" : y3_cat, "Footwear" : y4_cat, "y5" : y5_cat, "y6" : y6_cat, "Personal Care" : y7_cat, "Sporting Goods" : y8_cat}


    return full_df

def csv_matcher(csv_path = os.getenv("STYLES_CSV_PATH"), save_path = os.getenv("FILTERED_CSV_PATH"), images_path = os.getenv("RESIZED_REGISTRY_PATH")):
    df1 = pd.read_csv(csv_path, error_bad_lines = False)
    list_id = list(df1.id)
    images_from_csv = []
    for file in list_id:
        images_from_csv.append(str(file) + ".jpg")
    images_from_images = []
    for filename in os.listdir(images_path):
        images_from_images.append(filename)


    mismatched = list(set(images_from_images)^set(images_from_csv))

    for filename in os.listdir(images_path):
        if filename in mismatched:
            os.remove(os.path.join(images_path, filename))



    for filename in images_from_images:
        if filename in mismatched:
            images_from_images.remove(filename)

    images_from_images_stripped = []

    for filename in images_from_images:
        images_from_images_stripped.append(filename.replace(".jpg", ""))


    lista_id_int=[int(x) for x in images_from_images_stripped]
    filtered_df_styles = df1[df1['id'].isin(sorted(lista_id_int))]
    filtered_df_styles.to_csv(save_path)
    print(f"CSV Filtered and saved to {save_path}")
    return None


def x_loader(images_path = os.getenv("RESIZED_IMAGES_PATH"), csv_path = os.getenv("FILTERED_CSV_PATH")):
    df1 = pd.read_csv(csv_path, error_bad_lines = False)
    list_id = list(df1.id)
    images = []
    for file in list_id:
        images.append(str(file) + ".jpg")

    filenames = []
    for image in images:
        filenames.append((os.path.join(images_path, image)))

    images = []

    for i in range(0, len(filenames)):
        batch_filenames = filenames[i]
        images_batch = [np.array(Image.open(image_filename)) for image_filename in batch_filenames]
        images.extend(images_batch)
    return images
