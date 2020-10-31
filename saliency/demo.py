
#!usr/bin/env/python3

import os
import sys

sys.path.insert(0,'/home/qiang/QiangLi/Python_Utils_Functional/FixaTons-master/')

import FixaTons
# Put the dataset into the same fold with code
dataset = 'MIT1003'

'''
FixaTons.info.datasets()
FixaTons.info.stimuli('MIT1003')
FixaTons.info.subjects('MIT1003', 'i05june05_static_street_boston_p1010764.jpeg')

DATASET_NAME = 'MIT1003'
STIMULUS_NAME = 'i05june05_static_street_boston_p1010764.jpeg'
SUBJECT_ID = 'ya'

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
'''
#For all image in that dataset
for image in FixaTons.info.stimuli(dataset):
    for subject in FixaTons.info.subjects(dataset, image):
        FixaTons.show.scanpath(dataset, image, subject, animation=True, plotMaxDim=1000, wait_time=5000) 
