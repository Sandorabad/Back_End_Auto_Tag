from automatic_tagging.funciones.f_tagger import tagger, cat_fortag
import numpy as np
cat = cat_fortag()


for t, i in enumerate(cat.keys()):
    print(t , ":", cat[str(i)])
