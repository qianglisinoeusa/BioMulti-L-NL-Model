from __future__ import absolute_import, print_function, division, unicode_literals

import os
import sys
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass
from scipy.ndimage import zoom
from copy import copy
from skimage.transform import resize
import cv2
import random
import utils
from  dataset_path import COLLECTION_PATH
import math  
from imageio import imread

def get_imlist(path):
  """  
  Returns a list of filenames for all jpg images in a directory. """

  return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpeg')]

def save_saliency_map(image, saliency_map, filename):
    """ 
    Save saliency map on image.
    
    Args:
        image: Tensor of size (3,H,W)
        saliency_map: Tensor of size (1,H,W) 
        filename: string with complete path and file extension
    """

    saliency_map = saliency_map - saliency_map.min()
    saliency_map = saliency_map / saliency_map.max()
    saliency_map = saliency_map.clip(0,1)

    saliency_map = np.uint8(saliency_map * 255)
    saliency_map = cv2.resize(saliency_map, (682, 1024))

    image = cv2.resize(image, (682, 1024))

    # Apply JET colormap
    color_heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
    
    # Combine image with heatmap
    img_with_heatmap = np.float32(color_heatmap) + np.float32(image)
    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)

    cv2.imwrite(filename, np.uint8(255 * img_with_heatmap))


def stimu(DATASET_NAME):

    ''' This functions lists the names of the stimuli of a specified dataset '''

    return os.listdir(COLLECTION_PATH+'/'+DATASET_NAME+'/STIMULI')


if __name__ == '__main__':
    filesave = '/home/qiang/QiangLi/Python_Utils_Functional/FixaTons/mit1003/WECSF/color_saliency_map/'
    DATASET_NAME = 'mit1003' 
    for image in stimu(DATASET_NAME):
        stimulus = utils.stimulus(DATASET_NAME, image)
        saliency_map = utils.saliency_map_ql(DATASET_NAME, image)
        filename = os.path.join(filesave, image)
        save_saliency_map(stimulus, saliency_map, filename)
    print('done')
     
    
