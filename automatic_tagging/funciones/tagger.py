import numpy as np
import pandas as pd

import os
def cat_fortag(styles_folder = os.getenv("STYLES_CSV_PATH")):
    df_styles = pd.read_csv(styles_folder, error_bad_lines=False)

#MASTER
    labels1 = np.unique(df_styles.masterCategory).tolist()
#ACCESORIES
    labels2 = np.unique(df_styles[df_styles['masterCategory'] == labels1[0]].subCategory).tolist()
#APPAREL
    labels3 = np.unique(df_styles[df_styles['masterCategory'] == labels1[1]].subCategory).tolist()
#FOOTWEAR
    labels4 = np.unique(df_styles[df_styles['masterCategory'] == labels1[2]].subCategory).tolist()
#
    labels5 = np.unique(df_styles[df_styles['masterCategory'] == labels1[3]].subCategory).tolist()
#
    labels6 = np.unique(df_styles[df_styles['masterCategory'] == labels1[4]].subCategory).tolist()
#PERSONAL CARE
    labels7 = np.unique(df_styles[df_styles['masterCategory'] == labels1[5]].subCategory).tolist()
#SPORTING GOODS
    labels8 = np.unique(df_styles[df_styles['masterCategory'] == labels1[6]].subCategory).tolist()

    ret = {"master" : labels1, "accesories" : labels2, "apparel" : labels3, "footwear" : labels4, "personal_care" : labels7, "sporting_goods" : labels8}

    return ret




def tagger(predictions, categories):
    """Esta funcion toma varias predicciones (arrays de probabilidades) y retorna las correspondientes categorias"""

    cat_list = []
    for prediction in predictions:
        max_pos = np.argmax(prediction)
        max_cat = categories[max_pos]

        cat_list.append(max_cat)

    return cat_list
