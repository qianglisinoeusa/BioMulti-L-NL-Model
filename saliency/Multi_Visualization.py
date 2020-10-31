
#!usr/bin/env python3
# -*- coding: utf-8 -*-
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


def datasets():

    ''' This functions lists the names of the datasets included in the collection '''

    return os.listdir(COLLECTION_PATH)

def stimu(DATASET_NAME):

    ''' This functions lists the names of the stimuli of a specified dataset '''

    return os.listdir(COLLECTION_PATH+'/'+DATASET_NAME+'/STIMULI')

def subject(DATASET_NAME, STIMULUS_NAME):
    ''' This functions lists the names of the subjects which have been watching a
        specified stimuli of a dataset '''
    file_name, _ = os.path.splitext(STIMULUS_NAME)

    return os.listdir(
        os.path.join(
            COLLECTION_PATH,
            DATASET_NAME,
            'SCANPATHS',
            file_name
        )
    )

def show_scanpath(DATASET_NAME, STIMULUS_NAME, subject = 0, 
                       animation = True, wait_time = 500,
                       putLines = True, putNumbers = True, 
                       plotMaxDim = 1024):

    ''' This functions uses cv2 standard library to visualize the scanpath
        of a specified stimulus.

        By default, one random scanpath is chosen between available subjects. For
        a specific subject, it is possible to specify its id on the additional
        argument subject=id.

        It is possible to visualize it as an animation by setting the additional
        argument animation=True.

        Depending on the monitor or the image dimensions, it could be convenient to
        resize the images before to plot them. In such a case, user could indicate in
        the additional argument plotMaxDim=500 to set, for example, the maximum
        dimension to 500. By default, images are not resized.'''

    path = '/home/qiang/QiangLi/Python_Utils_Functional/FixaTons/MIT1003/ScanPath_Result/'    
    stimulus = utils.stimulus(DATASET_NAME, STIMULUS_NAME)

    scanpath = utils.scanpath(DATASET_NAME, STIMULUS_NAME, subject)

    toPlot = [stimulus,] # look, it is a list!

    for i in range(np.shape(scanpath)[0]):

        fixation = scanpath[i].astype(int)

        frame = np.copy(toPlot[-1]).astype(np.uint8)

        cv2.circle(frame,
                   (fixation[0], fixation[1]),
                   10, (0, 0, 255), 3)
        if putNumbers:
            cv2.putText(frame, str(i+1),
                        (fixation[0], fixation[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,0,0), thickness=2)
        if putLines and i>0:
            prec_fixation = scanpath[i-1].astype(int)
            cv2.line(frame, 
                    (prec_fixation[0], prec_fixation[1]), (fixation[0], fixation[1]),
                    (0, 0, 255), thickness = 1, lineType = 8, shift = 0)

        # if animation is required, frames are attached in a sequence
        # if not animation is required, older frames are removed
        toPlot.append(frame)
        if not animation:
            toPlot.pop(0)

    # if required, resize the frames
    if plotMaxDim:
        for i in range(len(toPlot)):
            h, w, _ = np.shape(toPlot[i])
            h, w = float(h), float(w)
            if h > w:
                w = (plotMaxDim / h) * w
                h = plotMaxDim
            else:
                h = (plotMaxDim / w) * h
                w = plotMaxDim
            h, w = int(h), int(w)
            toPlot[i] = cv2.resize(toPlot[i], (w, h), interpolation=cv2.INTER_CUBIC)

    for i in range(len(toPlot)):

        cv2.imshow('Scanpath of '+subject+' watching '+STIMULUS_NAME,
                   toPlot[i])
        if i == 0:
            milliseconds = 1
        elif i == 1:
            milliseconds = scanpath[0,3]
        else:
            milliseconds = scanpath[i-1,3] - scanpath[i-2,2]
        milliseconds *= 1000

        cv2.waitKey(int(milliseconds))
    cv2.imwrite(os.path.join(path ,STIMULUS_NAME), toPlot[-1])
    cv2.waitKey(wait_time)

    cv2.destroyAllWindows()

def show_map(DATASET_NAME, STIMULUS_NAME, showFixMap=False, showSalMap=True, showqlMap = False, wait_time=500, plotMaxDim=0):

    ''' 
    This functions uses cv2 standard library to visualize a specified
    stimulus. By default, stimulus is shown with its saliency map aside. It is possible
    to deactivate such option by setting the additional argument showSalMap=False.
    It is possible to show also (or alternatively) the fixation map by setting
    the additional argument showFixMap=True.
    Depending on the monitor or the image dimensions, it could be convenient to
    resize the images before to plot them. In such a case, user could indicate in
    the additional argument plotMaxDim=500 to set, for example, the maximum
    dimension to 500. By default, images are not resized.
    '''
    path = '/home/qiang/QiangLi/Python_Utils_Functional/FixaTons/MIT1003/Result/'    
    stimulus = utils.stimulus(DATASET_NAME, STIMULUS_NAME)
    print(STIMULUS_NAME)
    toPlot = stimulus[:400,:500]


    if showFixMap:
        fixation_map = utils.fixation_map(DATASET_NAME, STIMULUS_NAME)
        fixation_map = cv2.cvtColor(fixation_map,cv2.COLOR_GRAY2RGB)*255
        toPlot = np.concatenate((toPlot,fixation_map[:400,:500]), axis=1)

    if showSalMap:
        saliency_map = utils.saliency_map(DATASET_NAME, STIMULUS_NAME)
        saliency_map = cv2.cvtColor(saliency_map,cv2.COLOR_GRAY2RGB)
        toPlot = np.concatenate((toPlot,saliency_map[:400,:500]), axis=1)

    if showqlMap:
        saliency_map_ql = utils.saliency_map_ql(DATASET_NAME, STIMULUS_NAME)
        saliency_map_ql = cv2.cvtColor(saliency_map_ql, cv2.COLOR_GRAY2RGB)
        toPlot = np.concatenate((toPlot,saliency_map_ql[:400,:500]), axis=1)
    
   
    if plotMaxDim:
            h, w, _ = np.shape(toPlot)
            h, w = float(h), float(w)
            if h > w:
                w = (plotMaxDim / h) * w
                h = plotMaxDim
            else:
                h = (plotMaxDim / w) * h
                w = plotMaxDim
            h, w = int(h), int(w)
            toPlot = cv2.resize(toPlot, (w, h), interpolation=cv2.INTER_CUBIC)

    cv2.imshow(STIMULUS_NAME, toPlot)
    cv2.imwrite(os.path.join(path ,STIMULUS_NAME), toPlot)   
    cv2.waitKey(wait_time)
    cv2.destroyAllWindows()

if __name__ == '__main__':

    DATASET_NAME = 'MIT1003'
    
    for image in stimu(DATASET_NAME):
        show_map(DATASET_NAME, image,  showFixMap = False, showSalMap = True, showqlMap = True,
                wait_time=500, plotMaxDim = 1600)

