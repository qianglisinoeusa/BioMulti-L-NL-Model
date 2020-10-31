'''
and some influential models
    MIT1003
    MIT300
    CAT2000
    Toronto
    Koehler
    iSUN
    SALICON (both the 2015 and the 2017 edition and each with both the original mouse traces and the inferred fixations)
    FIGRIM
    OSIE
    NUSEF (the part with public images)

and some influential models:

    AIM
    SUN
    ContextAwareSaliency
    BMS
    GBVS
    GBVSIttiKoch
    Judd
    IttiKoch
    RARE2012
    CovSal

'''
from __future__ import print_function
import pysaliency
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

import pysaliency
import pysaliency.external_datasets

data_location = 'IttiKoch_datasets'
mit_stimuli, mit_fixations = pysaliency.external_datasets.get_mit1003(location=data_location)

model_location = 'IttiKoch_model'

Itti = pysaliency.IttiKoch(location=model_location)
saliency_map = Itti.saliency_map(mit_stimuli.stimuli[0])

plt.imshow(saliency_map)
plt.savefig('/home/qiang/QiangLi/Python_Utils_Functional/FixaTons/MIT1003/IttiKoch_Saliency_Map/saliency_map.png')
