#!usr/bin/env bash

# Copyright(c) 2020 QiangLi
# All Rights Reserved.
# qiang.li@uv.es
# Distributed under the (new) BSD License.


# Install pytorch
# The help link video: https://www.youtube.com/watch?v=15dzeLMFPC4.
# Pytorch only works under the virtual environment.

#   option1: source ~/.bashrc : (base) this conda environment.  
#            conda deactivate : quit conda environment. 
#   option2: mkdir virutal environment.
#            cd virutal environment.
#            python3 -m venv pytorch-BioMulti-L-NL  
#            -------------------------------------------------
#            source pytorch-BioMulti-L-NL/bin/activate
#            deactivate 
#            
sudo pip3 install torch==1.3.0+cpu torchvision==0.4.1+cpu  -f https://download.pytorch.org/whl/torch_stable.html

# Install option: (1) under conda environment- conda install  (2) venv-virutal environment-pip3 install
sudo pip3 install numpy
sudo pip3 install NeuroTools
sudo pip3 install --user git+https://github.com/NeuralEnsemble/MotionClouds.git
# Option: download then add path in local.
#sudo pip3 install MotionClouds
#conda install MotionClouds
sudo pip3 install statsmodels==0.10.0rc2 --pre
sudo pip3 install pyrtools
# Install from source
# git clone https://github.com/LabForComputationalVision/pyrtools
# cd pyrtools
# python setup.py -e 
# cd ..
sudo pip3 install PyWavelets
#conda install pywavelets

sudo pip3 install colour-science
#sudo pip3 install colorama
sudo pip3 install tensorflow
#conda install tensorflow
#conda install scipy
#conda install scikit-image
#conda install -c conda-forge opencv

sudo pip3 install scipy
#sudo python -m pip3 install scipy
#sudo apt-get install python3-scipy
sudo pip3 install opencv-python
sudo pip3 install sliding_window

# numpy
# scipy
# NeuroTools
# matplotlib
# IPython
# watermark
# version_information
# vispy
# pyglet
# imageio
# pyopengl


echo '===============================' \n
echo 'dependent toolbox install done' 



############################################################################
# Change gcc and g++ version
############################################################################
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 7
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 7
