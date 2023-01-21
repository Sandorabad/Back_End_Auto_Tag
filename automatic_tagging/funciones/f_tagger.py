import numpy as np
import pandas as pd

import os
def cat_for_tag():
    #MASTER
    labels1 = ['Accessories', 'Apparel', 'Footwear', 'Personal Care']

    #ACCESORIES
    labels2 = ['Accessories', 'Bags', 'Belts', 'Cufflinks', 'Eyewear', 'Gloves', 'Headwear', 'Jewellery', 'Mufflers',
                'Perfumes', 'Scarves', 'Shoe Accessories', 'Socks', 'Sports Accessories', 'Stoles', 'Ties', 'Umbrellas',
                'Wallets', 'Watches', 'Water Bottle']

    #APPAREL
    labels3 = ['Apparel Set', 'Bottomwear', 'Dress', 'Innerwear', 'Loungewear and Nightwear', 'Saree', 'Socks', 'Topwear']

    #FOOTWEAR
    labels4 = ['Flip Flops', 'Sandal', 'Shoes']

    #PERSONAL CARE
    labels5 = ['Bath and Body', 'Beauty Accessories', 'Eyes', 'Fragrance', 'Hair', 'Lips', 'Makeup', 'Nails', 'Perfumes',
                'Skin', 'Skin Care']

    cat_dict = {"Master" : labels1,
                "Accesories" : labels2,
                "Apparel" : labels3,
                "Footwear" : labels4,
                "Personal Care" : labels5}

    return cat_dict


def tagger(predictions, category):
    """Esta funcion toma varias predicciones (arrays de probabilidades) y retorna las correspondientes categorias"""
    cat_dict = cat_for_tag()

    if category == 'Master':
        cat_list = cat_dict['Master']
    elif category == 'Accesories':
        cat_list = cat_dict['Accesories']
    elif category == 'Apparel':
        cat_list = cat_dict['Apparel']
    elif category == 'Footwear':
        cat_list = cat_dict['Footwear']
    elif category == 'Personal Care':
        cat_list = cat_dict['Personal Care']
    else:
        print('Something went wrong')

    for prediction in predictions:
        max_pos = np.argmax(prediction)
        pred_category = cat_list[max_pos]

    return pred_category
