'''
========================================================================
Image 3D Visualization
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
 
Source:https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html

Install MATLAB Engine API for Python

Before you install, verify your Python and MATLAB configurations.

Check that your system has a supported version of Python and MATLAB R2014b or later. 
To check that Python is installed on your system, run Python at the operating system prompt.

Add the folder that contains the Python interpreter to your path, if it is not already there.

Find the path to the MATLAB folder. Start MATLAB and type matlabroot in the command window. 
Copy the path returned by matlabroot.

-----------------------------------------
[1] matlabroot
-----------------------------------------

To install the engine API, choose one of the following.

1. At a Windows operating system prompt —

----------------------------------------
[1] cd "matlabroot\extern\engines\python"
[2] python setup.py install
----------------------------------------

2. You might need administrator privileges to execute these commands.

At a macOS or Linux operating system prompt —

----------------------------------------
[1] cd "matlabroot/extern/engines/python"
[2] python setup.py install
----------------------------------------
You might need administrator privileges to execute these commands.

At the MATLAB command prompt —

------------------------------------------------------
[1] cd (fullfile(matlabroot,'extern','engines','python'))
[2] system('python setup.py install')
------------------------------------------------------
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
I = rgb2gray(I)
I = resize(I, (256, 256))

s2 = eng.mat_image_3D(matlab.double(I.tolist()), 2, 'jet', 'gray')

while eng.isvalid(s2):
    pass








