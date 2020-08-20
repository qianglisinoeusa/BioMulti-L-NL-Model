
#!usr/bin/env python3
# -*- coding: utf-8 -*-

'''
======================================================================
Copyright(c) 2020  Qiang Li
All Rights Reserved.
qiang.li@uv.es
Distributed under the (new) BSD License.
######################################################################
----------------------------------------------------------------------
Permission to use, copy, or modify this software and its documentation
for educational and research purposes only and without fee is here
granted, provided that this copyright notice and the original authors'
names appear on all copies and supporting documentation. This program
shall not be used, rewritten, or adapted as the basis of a commercial
software or hardware product without first obtaining permission of the
authors. The authors make no representations about the suitability of
this software for any purpose. It is provided "as is" without express
or implied warranty.
'''
import numpy as np
import matplotlib.pyplot as plt 
from scipy import io

mos = io.loadmat('/home/qiang/QiangLi/Python_Utils_Functional/FirstVersion-BioMulti-L-NL-Model-ongoing/TID2008/TID2008.mat')
print(len((mos['tid_MOS'])))
tid_MOS = mos['tid_MOS']
tid_MOS = 100-11*tid_MOS;


indices = np.load('/home/qiang/QiangLi/Python_Utils_Functional/BioMulti-L-NL-Model/database/indices.npy')

dr = np.load('/home/qiang/QiangLi/Python_Utils_Functional/BioMulti-L-NL-Model/database/drq.npy')
dw = np.load('/home/qiang/QiangLi/Python_Utils_Functional/BioMulti-L-NL-Model/database/dwq.npy')
dwf = np.load('/home/qiang/QiangLi/Python_Utils_Functional/BioMulti-L-NL-Model/database/dwfq.npy')
dwfs = np.load('/home/qiang/QiangLi/Python_Utils_Functional/BioMulti-L-NL-Model/database/dwfsq.npy')
dwfsn = np.load('/home/qiang/QiangLi/Python_Utils_Functional/BioMulti-L-NL-Model/database/dwfsnq.npy')

indi = indices[0][0] != 0  
#SB = tid_MOS[indices[indi:]].T
#print(SB)

SB = tid_MOS[indices.T[3]]
##################################################
#Visualization 
##################################################
figure.add_subplot(111, frame_on=False)
plt.figure(figsize=(12,6), dpi=94)
plt.subplots_adjust(wspace=0.5, hspace=0)
plt.margins(0,0)
plt.tick_params(labelcolor="none", bottom=False, left=False)

OB = dr[indi]                
#metric_1 = np.corrcoef(SB, OB.T) 
plt.subplot(151)
plt.scatter(SB, OB, c = 'g', cmap='hsv')
#plt.title('r = {}'.format(metric_1))

OB = dw[indi]                
#metric_2 = np.corrcoef(SB, OB.T)  
plt.subplot(152)
plt.scatter(SB, OB, c = 'g', cmap='hsv')
#plt.title('r = {}'.format(metric_2))

OB = dwf[indi]  
#metric_3 = np.corrcoef(SB, OB.T)  
plt.subplot(153)
plt.scatter(SB, OB, c = 'g', cmap='hsv')
#plt.title('r = {}'.format(metric_3))

OB = dwfs[indi]  
#metric_4 = np.corrcoef(SB, OB.T)  
plt.subplot(154)
plt.scatter(SB, OB, c = 'g', cmap='hsv')
#plt.title('r = {}'.format(metric_4))

OB = dwfsn[indi]  
#metric_5 = np.corrcoef(SB, OB.T)  
plt.subplot(155)
plt.scatter(SB, OB, c = 'g', cmap='hsv')
#plt.title('r = {}'.format(metric_5))

plt.xlabel('Mean Opinion Score')
plt.ylabel('Predict Scores')
plt.title('Model Evaluate with TID2008')

plt.show()