#-------------------------------------------------------------------------------
# Name:        main
# Purpose:     Testing the package pySaliencyMap
#
# Author:      Akisato Kimura <akisato@ieee.org>
#
# Created:     May 4, 2014
# Copyright:   (c) Akisato Kimura 2014-
# Licence:     All rights reserved
#-------------------------------------------------------------------------------

import os
import cv2
import matplotlib.pyplot as plt
import pySaliencyMap
from tqdm import trange


# main
if __name__ == '__main__':
    img_path = '/home/qiang/QiangLi/Python_Utils_Functional/FixaTons/MIT1003/STIMULI'
    #path1 = '/home/qiang/QiangLi/Python_Utils_Functional/BioMulti-L-NL-Model/saliency/Saliency_other_models/IttiKoch_model/IttiKoch_Result'
    path2 = '/home/qiang/QiangLi/Python_Utils_Functional/BioMulti-L-NL-Model/saliency/Saliency_other_models/AIM_model/AIM_Result'
    
    for filename in os.listdir(img_path):
        #print(filename)
        img = cv2.imread(os.path.join(img_path, filename))
        #cv2.imshow(filename, img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        img = cv2.resize(img, (682, 682))
        imgsize = img.shape
        img_width  = imgsize[1]
        img_height = imgsize[0]
        
        sm = pySaliencyMap.pySaliencyMap(img_width, img_height)
        saliency_map = sm.SMGetSM(img)
        #cv2.imshow(filename, saliency_map)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        cv2.imwrite(os.path.join(path1 ,filename), saliency_map*255) 
        