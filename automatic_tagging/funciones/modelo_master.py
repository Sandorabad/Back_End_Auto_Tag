## Importamos librerias necesarias
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras import Sequential, layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import optimizers
import keras
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

import pandas as pd
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

#modelos
from tensorflow.keras import Sequential, layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import optimizers


### Creamos nuestros objetos Generator

train_datagen = ImageDataGenerator(rescale = 1./255)
val_datagen = ImageDataGenerator(rescale = 1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

#creamos los generadores con batch de 30 fotos con el dir correpondiente
train_dir = "/content/drive/Shareddrives/auto-tag/data vale/Full/split1_full/split1_full/train"
val_dir = "/content/drive/Shareddrives/auto-tag/data vale/Full/split1_full/split1_full/val"
test_dir = "/content/drive/Shareddrives/auto-tag/data vale/Full/split1_full/split1_full/test"

train_generator = train_datagen.flow_from_directory(
                                train_dir,
                                target_size =(480, 360), # target_size = input image size
                                batch_size = 30, #para q steps per epoch sea 155
                                class_mode ='categorical')

val_generator = train_datagen.flow_from_directory(
                                val_dir,
                                target_size =(480, 360), # target_size = input image size
                                batch_size = 30, #para q steps per epoch sea 45
                                class_mode ='categorical')

test_generator = train_datagen.flow_from_directory(
                                test_dir,
                                target_size =(480, 360), # target_size = input image size
                                batch_size = 30,
                                class_mode ='categorical')




## Cargamos nuestro CSV
path_csv = "/content/drive/Shareddrives/auto-tag/data vale/csv/filtered_styles_full.csv"
df2 = pd.read_csv(path_csv, error_bad_lines = False)
df = df2



labels = np.unique(df.masterCategory).tolist()
num_cat = len(labels)
# Definimos y como el masterCategory y luego los convertimos a int para usar to_categorical
y = df[['masterCategory']]
y.masterCategory = pd.Categorical(y.masterCategory)
y_num = y.masterCategory.cat.codes
y_cat = to_categorical(y_num,num_cat)


print(len(y_cat[0]) == num_cat)

y_cat.shape



### Establecemos nuestro modelo

def load_model():
    """Cargamos el modelo pre-entrenado"""

    model = VGG16(weights="imagenet", include_top=False, input_shape = (480,360,3))

    return model

def set_nontrainable_layers(model):
    """Establecemos las primeras capas como no entrenables"""

    model.trainable = False

    return model

def add_last_layers(model):
    """Tomamos un modelo pre-entrenado, establecemos los parametros como no entrenables, y añadimos capas entrenables adicionales en la parte superior"""

    base_model = set_nontrainable_layers(model)
    flatten_layer = layers.Flatten()
    dense_layer = layers.Dense(500, activation='relu')
    prediction_layer = layers.Dense(num_cat, activation='softmax') #CUIDADO AQUI! USAMOS num_cat2 para que todo coincida

    model = models.Sequential([
        base_model,
        flatten_layer,
        dense_layer,
        prediction_layer
    ])

    return model

def build_model():
    """Traemos todas la funciones en una"""

    model = load_model()
    model = add_last_layers(model)
    model.compile(loss ='categorical_crossentropy',
                  optimizer = optimizers.Adam(learning_rate=1e-4),
                  metrics = ['accuracy'])
    return model


def train_model(model):

    history = model.fit(
             train_generator,
             steps_per_epoch = 50,
             epochs = 30,
             validation_data = val_generator,
             validation_steps = 20,callbacks = EarlyStopping(patience = 20,restore_best_weights = True))

	print("Model is trained")
    return model , history
