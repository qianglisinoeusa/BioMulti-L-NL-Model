
#!usr/bin/env/python3

import os
import sys

sys.path.insert(0,'/home/qiang/QiangLi/Python_Utils_Functional/FixaTons-master/')

import FixaTons
# Put the dataset into the same fold with code
dataset = 'SIENA12'

FixaTons.info.datasets()
FixaTons.info.stimuli('SIENA12')
FixaTons.info.subjects('SIENA12', 'land.jpg')

DATASET_NAME = 'SIENA12'
STIMULUS_NAME = 'land.jpg'
SUBJECT_ID = 'GT_10022017'

stimulus_matrix = FixaTons.get.stimulus(DATASET_NAME, STIMULUS_NAME)
saliency_map_matrix = FixaTons.get.saliency_map(DATASET_NAME, STIMULUS_NAME)
fixation_map_matrix = FixaTons.get.fixation_map(DATASET_NAME, STIMULUS_NAME)


print('Stimulus dims = ', stimulus_matrix.shape)
print('Saliency map dims =', saliency_map_matrix.shape)
print('Fixation map dims =', fixation_map_matrix.shape)

scanpath = FixaTons.get.scanpath(DATASET_NAME, STIMULUS_NAME, subject = SUBJECT_ID)
print(scanpath)
print("This scanpath has {} fixations.".format(len(scanpath)))


FixaTons.show.map(DATASET_NAME, STIMULUS_NAME, 
                  showSalMap = True, showFixMap = False,
                  wait_time=50000, plotMaxDim = 1024)

FixaTons.show.scanpath(DATASET_NAME, STIMULUS_NAME, subject= SUBJECT_ID, 
                       animation = True, wait_time = 0, 
                       putLines = True, putNumbers = True, 
                       plotMaxDim = 1024)


FixaTons.metrics.KLdiv(saliency_map_matrix, fixation_map_matrix)
FixaTons.metrics.AUC_Judd(saliency_map_matrix, fixation_map_matrix, jitter = True,toPlot = True)
FixaTons.metrics.NSS(saliencyMap_matrix, fixation_map_matrix)
#For all image in that dataset
#for image in FixaTons.list.stimuli(dataset):
    #Show the image aside its saliency map(5 seconds by default)
#    for subject in FixaTons.list.subjects(dataset, image):
        # Show the  correspondent  scanpath  as  an animation
        # (Look ,  time  of  exploration  in  the  animation  is  the
        # exact  time ,  from  the  dataset . 
#        FixaTons.show.scanpath(dataset, image, subject, animation=True, plotMaxDim=1000, wait_time=1000) 
