'''
========================================================================
Image 3D Stack Visualization
Copyright(c) 2020 QiangLi
All Right Reserved
qiang.li@uv.es
Distributed under the (new) BSD License
=======================================================================-
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

#If you don't open Matlab in your local machine, then you can start a new
#matlab environment 

import matlab.engine
#eng = matlab.engine.start_matlab()

# When there are multiple shared MATLAB sessions on your local machine, 
# connect to two different sessions one at a time by specifying their names.
names = matlab.engine.find_matlab()

# Connect to a shared MATLAB session that is already running on your local machine.
eng = matlab.engine.connect_matlab()
eng.addpath(r'/home/liqiang/CVP/BioMulti-L-NL-Model/')


import io
import matplotlib.pyplot as plt
from imageio import imread
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize, rescale

I = imread('/home/liqiang/CVP/BioMulti-L-NL-Model/imgs/clown.png')
I = resize(I, (256, 256, 3))

#s2 = eng.mat_image_3D(matlab.double(I.tolist()), 2, 'jet', 'gray')
s3 = eng.image_3D_stacked(matlab.double(I.tolist()))

while eng.isvalid(s3):
    pass
