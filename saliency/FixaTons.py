'''
@author: Dario Zanca, Ph.D.
@institutions: University of Siena

@e-mail: dariozanca@gmail.it
@tel: (+39) 333 82 78 072

@date: October, 2017
'''

#########################################################################################

import os

COLLECTION_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'FixaTons'
)

COLLECTION_PATH = os.path.join(
    '/home/qiang/QiangLi/Python_Utils_Functional/',
    'FixaTons'
)

'''
This file includes tools to an easy use of the collection of datasets. 
This tools help you in different tasks:
    - List information
    - Get data (matrices)
    - Visualize data
    - Compute metrics
'''

#########################################################################################

# IMPORT EXTERNAL LIBRARIES

import os
import cv2
import numpy as np

import _list_information_functions as info
import _get_data_functions as get
import _visualize_data_functions as show
import _visual_attention_metrics as metrics
import _compute_statistics as stats
