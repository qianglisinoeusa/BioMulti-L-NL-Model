#!usr/bin/env python3
# -*- coding: utf-8 -*-

'''
======================================================================
DigitialBrain Version 2.1
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

I would like to thank all of the open suorce contributors in the Python
open community, because open source make this script works. 

The *.ipynb version also can get when you test code.
'''

# Dependent toolbox 
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# pip3
#sudo pip3 install MotionClouds
#sudo pip3 install NeuroTools
#sudo pip3 install statsmodels==0.10.0rc2 --pre
#sudo pip3 install pyrtools
#sudo pip3 install PyWavelets
#sudo pip3 install colour-science
#sudo pip3 install colorama

# @caution: you need first run (InstallDependent.sh) for install necessary
# dependent toolbox

# Bash InstallDependent.sh

# Input the mudules and exter-packages
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import sys
sys.path.insert(0,'/home/qiang/QiangLi/Python_Utils_Functional/FirstVersion-BioMulti-L-NL-Model-ongoing/PyTorchSteerablePyramid')
#@caution: if you download via pip, you need to update some funciton:
#1). hand change scipy.msic to scipy.special otherwise you can not import out factorial function.
#2). add from imageio import imread in the utils script.
from steerable.SCFpyr_NumPy import SCFpyr_NumPy
import steerable.utils as utils
#from google.colab.patches import cv2_imshow
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab, rgb2xyz, rgb2yuv, yuv2rgb, gray2rgb, xyz2rgb, rgb2hsv
import glob
from imageio import imread
import math
from skimage.transform import resize, rescale
import scipy
import scipy.misc
from PIL import Image
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1 import ImageGrid
import cv2
import numpy as np
from numpy.linalg import inv
import time
from matplotlib import transforms
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
from skimage.util import dtype
import torch
import torch.nn.functional as F
import PIL.Image as pim
import warnings
from skimage import data, img_as_float
from skimage import exposure
from scipy import linalg
from mpl_toolkits import axes_grid1
from tqdm import tqdm_notebook as tqdm
from tqdm import trange

import pywt
import colour
from scipy import io
import sliding_window as sw
import matplotlib.gridspec as gridspec
from operator import sub
from mpl_toolkits import axes_grid1
from tqdm import tqdm_notebook as tqdm
from tqdm import trange
import scipy.ndimage

#%tensorflow_version 1x
#import tensorflow as tf
from pywt._doc_utils import wavedec2_keys, draw_2d_wp_basis
#from colorama import Fore

sys.path.insert(0, '/home/qiang/QiangLi/Python_Utils_Functional/FirstVersion-BioMulti-L-NL-Model-ongoing/MotionClouds-master/MotionClouds')
import MotionClouds as mc
import pyrtools as pt

#Sometimes, it will erros. Why? Go to check right postion use !ls
sys.path.insert(0, '/home/qiang/QiangLi/Python_Utils_Functional/FirstVersion-BioMulti-L-NL-Model-ongoing/SLIP/SLIP')
from SLIP import Image

#LogGabor can be install in the first time but when you second time install it,
#It will cause error. Here for how to fix it.
#!pip3 install --upgrade pip setuptools wheel
#!sudo apt-get install libpq-dev
sys.path.insert(0, '/home/qiang/QiangLi/Python_Utils_Functional/FirstVersion-BioMulti-L-NL-Model-ongoing/LogGabor/LogGabor')
from LogGabor import LogGabor
parameterfile = '/home/qiang/QiangLi/Python_Utils_Functional/FirstVersion-BioMulti-L-NL-Model-ongoing/LogGabor/default_param.py'
lg = LogGabor(parameterfile)
print('It works on 23 April2020')

#DWT wavelet filters 
sys.path.insert(0, '/home/qiang/QiangLi/Python_Utils_Functional/FirstVersion-BioMulti-L-NL-Model-ongoing/')

from nt_toolbox.general import *
from nt_toolbox.signal import *
from nt_toolbox.compute_wavelet_filter import *
print('It works on 27 April2020')

warnings.filterwarnings('ignore')
#%matplotlib inline
plt.style.use('ggplot')
plt.matplotlib.rcParams['font.size'] = 6
#%load_ext autoreload
#%autoreload 2
'''
############################################################################
#                          Visual Computing Saliency Map
############################################################################
# Copyright (c) 2020 QiangLi. 
# All rights reserved.
# Distributed under the (new) BSD License.
############################################################################
# Human visual inspired multi-layer LNL model. In this model, the main component
# are:
#    Nature Image --> VonKries Adaptation --> ATD  (Color processing phase)
#    Wavelets Transform --> Contrast sensivity function (CSF) --> Divisive
#    Normalization(DN)  --> Noise(Gaussian or Poisson)
#
# Evalute of model with TID2008 database.
#
# Redundancy redunction measure with Total Correlation(RBIG or Cortex module)
#
# This model derivated two version script： Matlab, Python. In the future, I
# want to implemented all of these code on C++ or Java. If our goal is simulate 
# of primate brain, we need to implement all everything to High performance
# Computer(HPC) with big framework architecture(C/C++/Java).
'''

############################################################################
# Function 
############################################################################
#---------------------------------------------------------------
# Papre image with right strucutre
#---------------------------------------------------------------
def prepare_colorarray(arr):
    """Check the shape of the array and convert it to
    floating point representation.
    """
    arr = np.asanyarray(arr)

    if arr.ndim != 3 or arr.shape[2] != 3:
        msg = "the input array must be have a shape == (.,.,3))"
        raise ValueError(msg)

    return dtype.img_as_float(arr)
#---------------------------------------------------------------
# Matrices that define conversion between different color spaces
# I will update more convert matrix at here.
#---------------------------------------------------------------
#rgb_to_xyz = np.array([[0.412453, 0.357580, 0.180423],
#              [0.212671, 0.715160, 0.072169],
#              [0.019334, 0.119193, 0.950227]])
#xyz_to_rgb = linalg.inv(rgb_to_xyz)

xyz_to_atd = np.array([[0.297, 0.72, -0.107],
            [-0.449, 0.29, -0.077],
            [0.086, -0.59, 0.501]])
atd_to_xyz = linalg.inv(xyz_to_atd)
atd_to_xyz_updated=np.array([[0.979, 1.189, 1.232],
              [-1.535, 0.764, 1.163],
              [0.445, 0.135, 2.079]])

srgb_to_iou=np.array([[0.2814, 0.6938, 0.0638],
          [-0.0971, 0.1458, -0.0250],
          [-0.0930, -0.2529, 0.4665]])
iou_to_srgb = linalg.inv(srgb_to_iou)

rgb_to_yuv=np.array([[0.299, 0.587, 0.114],
          [-0.147, -0.288, 0.436],
          [0.615, -0.515, -0.100]])

rgb_to_yiq=np.array([[0.299, 0.587, 0.114],
          [0.596, -0.274, -0.322],
          [0.211, -0.523, 0.312]])
#---------------------------------------------------------------
# Conversion between different color spaces
#---------------------------------------------------------------
def convert(matrix, arr):
    """Do the color space conversion.
    Parameters
    ----------
    matrix : array_like
        The 3x3 matrix to use.
    arr : array_like
        The input array.
    Returns
    -------
    out : ndarray, dtype=float
        The converted array.
    """
    arr = prepare_colorarray(arr)
    arr = np.swapaxes(arr, 0, 2)
    oldshape = arr.shape
    arr = np.reshape(arr, (3, -1))
    out = np.dot(matrix, arr)
    out.shape = oldshape
    out = np.swapaxes(out, 2, 0)

    return np.ascontiguousarray(out)

def xyz2atd(xyz):
    return convert(xyz_to_atd, xyz)
def atd2xyz(atd):
    return convert(atd_to_xyz, atd)
  
def atd2xyz_updated(atd):
    return convert(atd_to_xyz_updated, atd)
 
def linsrgb_to_srgb (linsrgb):
    gamma = 1.055 * linsrgb**(1./2.4) - 0.055
    scale = linsrgb * 12.92
    return np.where (linsrgb > 0.0031308, gamma, scale)

def srgb2iou(srgb):
    return convert(srgb_to_iou, srgb)

def iou2srgb(iou):
    return convert(iou_to_srgb,iou)

def rgb2yuv_matrix(rgb):
    return convert(rgb_to_yuv,rgb)

def rgb2yiq(rgb):
    return convert(rgb_to_yiq, rgb)
#---------------------------------------------------------------
# Check max and min value in each channel
#---------------------------------------------------------------
def plot_MinMax(X, labels=["R", "G", "B"]):
    print("-------------------------------")
    for i, lab in enumerate(labels):
        mi=np.min(X[:,:,i])*255
        ma=np.max(X[:,:,i])*255
        print("{} : MIN={:8.4f}, MAX={:8.4f}".format(lab,mi,ma))
    #test=np.array(images[2])    
    #plot_MinMax(test, labels=["R","G","B"])
#---------------------------------------------------------------
# Visualization the ATD opponent channel with presudo color
#---------------------------------------------------------------
def visual_channel_iou(image_iou, dim):
    """ 
    Opponent channel processing theory
    PresudoColor visualization
    The input array must be ATD color space
    """
    z=np.zeros(image_iou.shape)
    if dim != 0:
        print(np.mean(image_iou[:,:,0]))
        print(np.max(image_iou[:,:,0]))
        print(np.min(image_iou[:,:,0]))
        z[:,:,0]= np.min(image_iou[:,:,0])
        z[:,:,dim] = image_iou[:,:,dim]
        z = iou2srgb(z)
        z = (255*np.clip(z,0,1)).astype('uint8') 

    return(z)
#---------------------------------------------------------------
# Implemented Weber's and Fechner's law
#---------------------------------------------------------------
def waber(lumin, lambdaValue):
    """ 
    Weber's law nspired by Physhilogy experiment.
    """
    # lambdaValue normally select 0.6
    w = lumin**lambdaValue
    #w = (255*np.clip(w,0,1)).astype('uint8') 
    return(w)
#---------------------------------------------------------------
# Implemented Entropy 
#---------------------------------------------------------------
def entropy(img):
    """ 
    Calculate the entropy of image
    """
    hist, _ = np.histogram(img)
    hist = hist[hist > 0]
    return -np.log2(hist / hist.sum()).sum()

def show_entropy(band_name, img):    
    """ 
    Plot the entropy of image
    """
    bits = entropy(img)
    per_pixel = bits / img.size
    print(f"{band_name:3s} entropy = {bits:7.2f} bits, {per_pixel:7.6f} per pixel")

def image_entropy(img):
    """
    Calculate the entropy of an image-update funciton
    """
    histogram, _ = np.histogram(img)
    histogram_length = sum(histogram)

    samples_probability = [float(h) / histogram_length for h in histogram]

    return -sum([p * math.log(p, 2) for p in samples_probability if p != 0])

def print_entropy(band_name, img):
    """ 
    plot the entropy of image with each band name  
    """
    bits = image_entropy(img)
    #per_pixel = bits / img.size
    print(f"{band_name:3s} entropy = {bits:7.2f} bits")
#---------------------------------------------------------------
# Implemented mutual_information
#---------------------------------------------------------------
def mutual_information(hgram):
     """ 
     Mutual information for joint histogram
     """
     # Convert bins counts to probability values
     pxy = hgram / float(np.sum(hgram))
     px = np.sum(pxy, axis=1) # marginal for x over y
     py = np.sum(pxy, axis=0) # marginal for y over x
     px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
     # Now we can do the calculation using the pxy, px_py 2D arrays
     nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
     return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
#---------------------------------------------------------------
# Implemented resized_img function
#---------------------------------------------------------------
def resized_img(img):
    '''
    Resized function for image size with (256, 256)
    '''
    image_resized = resize(img, (256, 256, 3), anti_aliasing=True)
    return image_resized
#---------------------------------------------------------------
# Implemented visualization image histogram with 3D function
#---------------------------------------------------------------
def histogram3dplot(h, e, fig=None):
    '''
    Visualization 3D histogram of image. 
    '''  
    M, N, O = h.shape
    idxR = np.arange(M)
    idxG = np.arange(N)
    idxB = np.arange(O)

    R, G, B = np.meshgrid(idxR, idxG, idxB)
    a = np.diff(e[0])[0]
    b = a/2
    R = a * R + b

    a = np.diff(e[1])[0]
    b = a/2
    G = a * G + b

    a = np.diff(e[2])[0]
    b = a/2
    B = a * B + b

    colors = np.vstack((R.flatten(), G.flatten(), B.flatten())).T/255
    h = h / np.sum(h)
    if fig is not None:
        f = plt.figure(fig)
    else:
        f = plt.gcf()
    ax = f.add_subplot(111, projection='3d')     
    mxbins = np.array([M,N,O]).max()
    ax.scatter(R.flatten(), G.flatten(), B.flatten(), s=h.flatten()*(256/mxbins)**3/2, c=colors)
    #ax.set_xlabel('Red')
    #ax.set_ylabel('Green')
    #ax.set_zlabel('Blue')
#---------------------------------------------------------------
# Calculate correlation coffecient value for image split channel
#---------------------------------------------------------------
def show_coeff(band_name, img_ch, img_chs):
    '''
    Plot the correlation coefficient between two color channel
    '''
    coeff=np.corrcoef(img_ch.ravel(), img_chs.ravel())[0, 1]
    print(f"{band_name:3s} coeff = {coeff:.2f} coeff")
#---------------------------------------------------------------
# Visualization Lab colorspace with presudo-color
#---------------------------------------------------------------
def extract_single_dim_from_LAB_convert_to_RGB(image,dim):
    '''
    Visualization presudo color in the LAB color space.
    '''
    z = np.zeros(image.shape)
    if dim != 0:
        z[:,:,0]=30 ## I need brightness to plot the image along 1st or 2nd axis
    z[:,:,dim] = image[:,:,dim]
    z = lab2rgb(z)
    return(z)
#---------------------------------------------------------------
# Calculate MSE between raw image and reference image
#---------------------------------------------------------------
def mse(reference, query):
    '''
    Calculate the mse between origin image and query image
    '''
    (ref, que) = (reference.astype('double'), query.astype('double'))
    diff = ref - que
    square = (diff ** 2)
    mean = square.mean()
    return mean
#---------------------------------------------------------------
# Calculate PSNR between raw image and reference image
#---------------------------------------------------------------
def psnr(reference, query, normal=255):
    '''
        Calculate the PSNR of original image and query image.
    '''
    normalization = float(normal)
    msev = mse(reference, query)
    if msev != 0:
        value = 10.0 * np.log10(normalization * normalization / msev)
    else:
        value = float("inf")
    return value

def show_psnr(band_name, reference, query):
    '''
    plot psnr of reference image and query image
    '''
    psnr_ = psnr(reference, query, normal=255)
    print(f"{band_name:3s} psnr_ = {psnr_:.2f} psnr_")
#---------------------------------------------------------------
# Convert RGB image to gray
#---------------------------------------------------------------
def convert_gray(im_rgb):
    '''
    Convert image gray scale, image of shape (None,None,3)
    '''
    R=im_rgb[:,:,0]
    G=im_rgb[:,:,1]
    B=im_rgb[:,:,2]
    L = R*299/1000 + G*587/1000+B*114/1000

    return(L)

def convert_to_luminance(x):
    '''
    Convert color image into luminacne
    '''
    return np.dot(x[..., :3], [0.299, 0.587, 0.144]).astype('double')
#---------------------------------------------------------------
# Implemented CSF, you can adjust fourier space size and frequency
#---------------------------------------------------------------
def make_CSF(x, nfreq):
    '''
    Contrast Sensitivity Function implemented with Delay version.
    
    The CSF measures the sensitivity of human visual system to the various frequencies of visual stimuli, 
    Here we apply an adjusted CSF model given by:

    The mathmatic equation of CSF located in my CortexComputing notebook(Random section)

    Input:
        x - Define size of domain, float
        nfreq - Fourier frequency, float

    Output:
        CSF -  Fourier Space of CSF, 2darray
    '''
    #w=0.7
    #up_bound=7.8909
    #down_bound= 0.9809
    #a=2.6
    #b=0.0192
    #c=0.114
    #e=1.1
    params=[0.7, 7.8909, 0.9809, 2.6, 0.0192, 0.114, 1.1]

    N_x=nfreq
    N_x_=np.linspace(0, N_x, x+1)-nfreq/2
    N_x_up=N_x_[:-1]

    [xplane,yplane]=np.meshgrid(N_x_up, N_x_up)
    plane=(xplane+1j*yplane)  
    radfreq=np.abs(plane)	
    
    s=(1-params[0])/2*np.cos(4*np.angle(plane))+(1+params[0])/2
    radfreq=radfreq/s
    print(radfreq.shape)
    csf = params[3]*(params[4]+params[5]*radfreq)*np.exp(-(params[5]*radfreq)**params[6])
    f = radfreq < params[1]
    csf[f] = params[2]
    return csf
#---------------------------------------------------------------
# Implemented Gabor function with control contrast, luminacne
# velocity, orienttion and sigma of Gaussian function. 
#---------------------------------------------------------------
def Gabor(size, L0, c, f, theta, sigma, center=None):
    """ Make a Gabor functional with control parameters.
    :Input:
        -size: array, float
        -L0: luminance, float
        -c: contrast, float
        -theta: orientation, float
        -sigma: fwhm of gaussian, float
        -center: center coordinate of spatial map, float
    :output:
        -Gabor
    """
    x=np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x=y=int(size//2)
    else:
        x=center[0]
        y=center[1]
    [x,y]=np.meshgrid(range(-x,x),range(-y,y))  
    L=L0*(1.0+c*np.sin(2*np.pi*f*(x*np.cos(theta)+y*np.sin(theta))) * np.exp(-(x**2+y**2)/2*sigma**2))
    return L
#---------------------------------------------------------------
# Global contrast normalization
#---------------------------------------------------------------
def global_contrast_normalization(X, s, lmda, epsilon):  
    '''
        Global contrast normalization
    '''
    for i in range(len(X)):
        X_average = np.mean(X[i])
        print('Mean: ', X_average)
        x = X[i] - X_average
        # `su` is here the mean, instead of the sum
        contrast = np.sqrt(lmda + np.mean(x**2))
        x = s * x / max(contrast, epsilon)
        #global_contrast_normalization(images, 1, 10, 0.000000001)
#---------------------------------------------------------------
# Implemented gaussian filter
#---------------------------------------------------------------
def gaussian_filter(kernel_shape):
    '''
        Gaussian filter 
    '''
    x = np.zeros(kernel_shape, dtype='float32')

    def gauss(x, y, sigma=2.0):
        Z = 2 * np.pi * sigma ** 2
        return  1. / Z * np.exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))
    mid = np.floor(kernel_shape[-1] / 2.)
    for kernel_idx in range(0, kernel_shape[1]):
        for i in range(0, kernel_shape[2]):
            for j in range(0, kernel_shape[3]):
                x[0, kernel_idx, i, j] = gauss(i - mid, j - mid)
    return x / np.sum(x)
#---------------------------------------------------------------
# Local mean and local divisive
# local contrast normalization for increase feature
# which inspired by Divisive Normalization
#---------------------------------------------------------------
def DivisiveNormalization(image,radius=9):
    """
    image: torch.Tensor , .shape => (1,channels,height,width) 

    radius: Gaussian filter size (int), odd
    """
    if radius%2 == 0:
        radius += 1
    def get_gaussian_filter(kernel_shape):
        x = np.zeros(kernel_shape, dtype='float64')

        def gauss(x, y, sigma=2.0):
            Z = 2 * np.pi * sigma ** 2
            return  1. / Z * np.exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))

        mid = np.floor(kernel_shape[-1] / 2.)
        for kernel_idx in range(0, kernel_shape[1]):
            for i in range(0, kernel_shape[2]):
                for j in range(0, kernel_shape[3]):
                    x[0, kernel_idx, i, j] = gauss(i - mid, j - mid)

        return x / np.sum(x)

    n,c,h,w = image.shape[0],image.shape[1],image.shape[2],image.shape[3]

    gaussian_filter = torch.Tensor(get_gaussian_filter((1,c,radius,radius)))
    filtered_out = F.conv2d(image,gaussian_filter,padding=radius-1)
    mid = int(np.floor(gaussian_filter.shape[2] / 2.))
    ### Subtractive Normalization
    centered_image = image - filtered_out[:,:,mid:-mid,mid:-mid]

    ## Variance Calc
    sum_sqr_image = F.conv2d(centered_image.pow(2),gaussian_filter,padding=radius-1)
    s_deviation = sum_sqr_image[:,:,mid:-mid,mid:-mid].sqrt()
    per_img_mean = s_deviation.mean()

    ## Divisive Normalization
    divisor = np.maximum(per_img_mean.numpy(),s_deviation.numpy())
    divisor = np.maximum(divisor, 1e-4)
    new_image = centered_image / torch.Tensor(divisor)
    return new_image
#---------------------------------------------------------------
# Plot image and hist
#---------------------------------------------------------------
def plot_img_and_hist(image, axes, bins=256):
    """
    The script modifies from skimage. 
    Plot an image along with its histogram and cumulative histogram.
    """
    image = img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf
#---------------------------------------------------------------
# Adjust Gamma
#---------------------------------------------------------------
def adjust_gamma(image, gamma=0.8):  
    """The nonlinearlity convert between human visual and display screen""" 
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)
#---------------------------------------------------------------
# Visualization wavelet transform coefficient with verious scale
#---------------------------------------------------------------
def plot_wavelet_ql(fW, Jmin=0):
    """
        plot_wavelet - plot wavelets coefficients.

        U = plot_wavelet(fW, Jmin):
    """
    def rescaleWav(A):
        v = abs(A).max()
        B = A.copy()
        if v > 0:
            B = .5 + .5 * A / v
        return B
    
    n = fW[0,0].shape[1]
    Jmax = int(np.log2(n)) - 1
    U = fW[0,0].copy()
    for j in np.arange(Jmax, Jmin - 1, -1):
        U[:2 ** j:,    2 ** j:2 **
            (j + 1):] = rescaleWav(U[:2 ** j:, 2 ** j:2 ** (j + 1):])
        U[2 ** j:2 ** (j + 1):, :2 **
            j:] = rescaleWav(U[2 ** j:2 ** (j + 1):, :2 ** j:])
        U[2 ** j:2 ** (j + 1):, 2 ** j:2 ** (j + 1):] = (
            rescaleWav(U[2 ** j:2 ** (j + 1):, 2 ** j:2 ** (j + 1):]))
    U[:2 ** j:, :2 ** j:] = nt.rescale(U[:2 ** j:, :2 ** j:])
    imageplot(U)
    for j in np.arange(Jmax, Jmin - 1, -1):
        plt.plot([0, 2 ** (j + 1)], [2 ** j, 2 ** j], 'r')
        plt.plot([2 ** j, 2 ** j], [0, 2 ** (j + 1)], 'r')
    plt.plot([0, n], [0, 0], 'r')
    plt.plot([0, n], [n, n], 'r')
    plt.plot([0, 0], [0, n], 'r')
    plt.plot([n, n], [0, n], 'r')
    return U
#---------------------------------------------------------------
# Wavelet transform coefficient with Daubechies filter
# Visualization coefficients
#---------------------------------------------------------------
def wavelet_transform(I, Jmin=1, h = compute_wavelet_filter("Daubechies",6)) :
    """
    2D-Wavelet decomposition, using Mallat's algorithm.
    By default, the convolution filters are those proposed by
    Ingrid Daubechies in her landmark 1988-1992 papers.
    """
    wI = perform_wavortho_transf(I, Jmin, + 1, h)
    return wI

def iwavelet_transform(wI, Jmin=1, h = compute_wavelet_filter("Daubechies",6)) :
    """
    Invert the Wavelet decomposition by rolling up the operations above.
    """
    I = perform_wavortho_transf(wI, Jmin, - 1, h)
    return I

def display(im):  
    """
    Displays an image using the methods of the 'matplotlib' library.
    """
    plt.figure(figsize=(8,8))                   
    plt.imshow( im, cmap="gray", vmin=0, vmax=1)
    plt.axis("off") 

#---------------------------------------------------------------
# Wavelet threshold coefficient with Daubechies filter
# Visualization coefficients
#---------------------------------------------------------------
def Wavelet_threshold(wI, threshold) :   
    """
    Re-implement a thresholding routine and create a copy of the Wavelet transform
    Remove all the small coefficients then Invert the new transform
    """
    wI_thresh = wI.copy()                  
    wI_thresh[ abs(wI) < threshold ] = 0   
    I_thresh = iwavelet_transform(wI_thresh)  
    return I_thresh

def thresh_hard(u,t):
    """Hard threshold for Ortho-wavelets"""
    return u*(np.abs(u)>t)

def thresh_soft(u,t):
    """Soft threshold for Ortho-wavelets"""
    return np.maximum(1-t/np.abs(u), 0)*u

#---------------------------------------------------------------
# Visualization sp3_filter coefficients
#--------------------------------------------------------------
def make_grid_coeff_ql(coeff, normalize=True):
    '''
    Visualization function for building a large image that contains the
    low-pass, high-pass and all intermediate levels in the steerable pyramid. 
    For the complex intermediate bands, the real part is visualized.

    Args:
        coeff (list): complex pyramid stored as list containing all levels
        normalize (bool, optional): Defaults to True. Whether to normalize each band

    Returns:
        np.ndarray: large image that contains grid of all bands and orientations
    '''

    M, N = coeff[0,1].shape
    Norients = pyr.num_scales
    #out = np.zeros((M * 2 - coeff[3, 3].shape[0], Norients * N))
    out=np.zeros((258,514))
    currentx, currenty = 0, 0

    for i in range(1, pyr.num_scales):
        for j in range(4):
            tmp = coeff[i,j].real
            m, n = tmp.shape
            if normalize:
                tmp = 255 * tmp/tmp.max()
            tmp[m-1,:] = 255
            tmp[:,n-1] = 255
            out[currentx:currentx+m,currenty:currenty+n] = tmp
            currenty += n
        currentx += coeff[i, 0].shape[0]
        currenty = 0

    a, b = coeff[3,3].shape
    out[currentx: currentx+a, currenty: currenty+b] = 255 * coeff[3,3]/coeff[3,3].max()
    out[0,:] = 255
    out[:,0] = 255
    return out.astype(np.uint8)
#---------------------------------------------------------------
# Visualization entropy 2D map
#--------------------------------------------------------------
def entropy_2d(signal):
    '''
    function returns entropy of a signal
    signal must be a 1-D numpy array
    '''
    lensig=signal.size
    symset=list(set(signal))
    numsym=len(symset)
    propab=[np.size(signal[signal==i])/(1.0*lensig) for i in symset]
    ent=np.sum([p*np.log2(1.0/p) for p in propab])
    return ent
#---------------------------------------------------------------
# Add colorbar with right size
#--------------------------------------------------------------
def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)
#---------------------------------------------------------------
# View image and Fourier space in the same time and
# Plot the centerl response 
#--------------------------------------------------------------
def viewimage(img):
    '''
    In the current figure window, show the image and its
    central rows and columns (round((end+1)/2)), as well 
    as its amplitude spectrum and its central rows and columns.

    Input: 2d array (img)

    Output: 2d array (img and correspond its Fourier space)

    '''
    img2 = 20*np.log10(np.fft.fftshift(np.abs(np.fft.fft2(img))))
    plt.figure(figsize=(10, 10))
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.subplot(2,3,1)
    plt.imshow(img, cmap='gray'), plt.axis('off'), plt.title('image')
    plt.subplot(2,3,2)
    plt.plot(img[np.int64((img.shape[0]+1)/2) ,:], 'r.-' ) 
    plt.axis('off')
    plt.title('central row of image')
    plt.subplot(2,3,3)
    plt.plot(img[:, np.int64((img.shape[0]+1)/2)], 'r.-')
    plt.axis('off')
    plt.title('central column of image')
    plt.subplot(2,3,4)
    plt.imshow(img2, cmap='gray')
    plt.axis('off')
    plt.title('magnitude spectrum')
    plt.subplot(2,3,5)
    plt.plot(img2[:, np.int64((img2.shape[0]+1)/2)], 'b.-')
    plt.axis('off')
    plt.title('central row of magnitude spectrum')
    plt.subplot(2,3,6)
    plt.plot(img2[np.int64((img2.shape[0]+1)/2), :], 'b.-')
    plt.axis('off')
    plt.title('central column of magnitude spectrum')
    plt.tight_layout()
    plt.show()
#---------------------------------------------------------------
# Saturation is an element-wise exponential function.
#--------------------------------------------------------------
'''
def saturation_f(x,g,xm,epsilon,sizeT):
    
    SATURATION_F is an element-wise exponential function (saturation). 
    It is good to (1) model the saturation in Wilson-Cowan recurrent networks, 
    and (2) as a crude (fixed) approximation to the luminance-brightness transform. 

    This saturation is normalized and modified to have these two good properties:
    (a) Some specific input, xm (e.g. the median, the average) maps into itself: xm = f(xm).

        f(x) = sign(x)*K*|x|^g  , where the constant K=xm^(1-g)

    (b) Plain exponential is modified close to the origin to avoid the singularity of
    the derivative of saturating exponentials at zero.
    This problem is solved by imposing a parabolic behavior below a certain
    threshold (epsilon) and imposing the continuity of the parabola and its
    first derivative at epsilon.

        f(x) = sign(x)*K*|x|^g             for |x| > epsilon
                sign(x)*(a*|x|+b*|x|^2)     for |x| <= epsilon

    with:
                a = (2-g)*K*epsilon^(g-1)
                b = (g-1)*K*epsilon^(g-2)

    The derivative is (of course) signal dependent:

        df/dx = g*K*|x|^(g-1)   for |x| > epsilon   [bigger with xm and decreases with signal]
                a + 2*b*|x|     for |x| <= epsilon  [bigger with xm and decreases with signal (note that b<0)]

    In the end, the slope at the origin depends on the constant xm (bigger for bigger xm). 

    The program gives the function and the derivative. For the inverse see INV_SATURATION_F.M

    For vector/matrix inputs x, the vector/matrix with anchor points, xm, has to be the same size as x.

    USE:    [f,dfdx] = saturation_f(x,gamma,xm,epsilon);

    x     = n*m matrix with the values 
    gamma = exponent (scalar)
    xm    = n*m matrix with the anchor values (in wavelet representations typically anchors will be different for different subbands)
    epsilon = threshold (scalar). It can also be a matrix the same size as x (again different epsilons per subband, e.g. epsilon = 1e-3*x_average)

    
    K = tf.pow(xm, tf.scalar_mul(1 - g, tf.ones([sizeT,1])))
    K = tf.where(tf.math.is_nan(K), tf.zeros_like(K), K)

    a = (2 - g) * K*(epsilon**(g - 1))
    a = tf.where(tf.math.is_nan(a), tf.zeros_like(a), a)
    b = (g-1) * K*(epsilon**(g - 2))
    b = tf.where(tf.math.is_nan(b), tf.zeros_like(b), b)

    pG = tf.math.greater(x, tf.ones([sizeT,1]) * epsilon)
    pG_zeros = tf.count_nonzero(pG)

    pp1 = tf.math.less_equal(x, tf.ones([sizeT,1]) * epsilon)
    pp2 = tf.math.greater_equal(x, tf.zeros([sizeT,1]))
    pp1_zeros = tf.count_nonzero(pp1)
    pp2_zeros = tf.count_nonzero(pp2)

    nG = tf.math.less(x, -tf.ones([sizeT,1]) * epsilon)
    np1 = tf.math.greater(x, -tf.ones([sizeT,1]) * epsilon)
    np2 = tf.math.less_equal(x, tf.zeros([sizeT,1]))
    nG_zeros = tf.count_nonzero(nG)
    np1_zeros = tf.count_nonzero(np1)
    np2_zeros = tf.count_nonzero(np2)

    f = x

    def f1(): return tf.where(pG, K*tf.pow(x, (g * tf.ones([sizeT,1]))), f)
    def f2(): return f
    f1 = tf.cond(tf.math.greater(pG_zeros, 0), f1, f2)

    def f3(): return tf.where(nG, -K*tf.pow(tf.abs(x), (g * tf.ones([sizeT,1]))), f1)
    def f4(): return f1
    f2 = tf.cond(tf.math.greater(nG_zeros, 0), f3, f4)

    def f5(): return tf.where(tf.math.logical_and(pp1,pp2), a * tf.abs(x) + b *tf.pow(x, 2 * tf.ones([sizeT,1])), f2)
    def f6(): return f2
    f3 = tf.cond(tf.math.greater(pp1_zeros + pp2_zeros, 1), f5, f6)

    def f7(): return tf.where(tf.math.logical_and(np1,np2), -(a * tf.abs(x) + b * tf.pow(x, 2 * tf.ones([sizeT,1]))), f3)
    def f8(): return f3
    f4 = tf.cond(tf.math.greater(np1_zeros + np2_zeros, 1), f7, f8)

    return f4
'''
def im2col_sliding_strided(img, block_size, stepsize=1):
    '''
    Convert image matrix into column

    Input:
    ----------
    img: 2dArray, float
    block_size: 2dArray, float
    stepsize: default, 1

    Output:
    -----------
    Column 
    '''
    m,n = img.shape
    s0, s1 = img.strides    
    nrows = m-block_size[0]+1
    ncols = n-block_size[1]+1
    shp = block_size[0], block_size[1], nrows, ncols
    strd = s0, s1, s0, s1

    out_view = np.lib.stride_tricks.as_strided(img, shape=shp, strides=strd)
    return out_view.reshape(block_size[0]*block_size[1],-1)[:,::stepsize]

def resp_norm_stage_lin_nonlin(x,Hn,Hd,a,b,jacob):
    '''    
    RESP_NORM_STAGE_lin_nonlin is the same as RESP_NORM_STAGE but it returns 
    not only the final nonlinear output, but also the intermediate linear stage. 
    RESP_NORM_STAGE computes the responses and Jacobian of a substractive/divisive neural stage
     
    The response vector "y" given the input vector "x" is
    
                     x - a*Hn*x
            y = ----------------------
                    b + Hd*abs(x)
    
    And each element of the Jacobian matrix is:
    
                            delta_{ij} - a*Hn_{ij}             (x_i - \sum_k a*H_{ik}*x_k) * Hd_{ij}
      nabla_R_{ij} =  ----------------------------------  -  -------------------------------------------
                         b + \sum_k Hd_{ik} abs(x_k)           ( b + \sum_k Hd_{ik} abs(x_k) )^2
    
     [ y , y_interm, nablaL, nablaR, sx ] = resp_norm_stage(x,Hn,Hd,a,b,comp_jacob);
    
    
           x = input vector
    
           Hn and Hd = interaction kernels at the numerator and denominator
                       (see make_conv_kernel for Gaussian kernels).
                       The sum of the rows in these kernels is normalized to one        
    
           a = norm of the rows in the substractive interaction
    
           b = saturation constant in the divisive normalization
    
           if comp_jacobian == 1 -> computes the Jacobian (otherwise it only
           computes response).
    
    '''
    sx = np.sign(x)
    
    num = x - a*Hn*x
    div = b + Hd*np.abs(x)

    nablaL = np.eye(len(x)) - a*Hn

    # xx = x - a*Hn*x;
    y = np.divide(num,div)

    # Jacobian (if required)

    if jacob ==1:
        N=len(x)
        nablaR = np.zeros(N,N)
        div2 = div**2 
        for i in range(1,N):
            one = np.zeros(1,N)
            one[i] = 1
            nablaR[i,:] = (1/div[i]) * (one - a*Hn[i,:]) - (num[i]/div2[i]) * Hd[i,:]
    else:
        nablaR =0 

def normalize_saliency_map(saliency_map, cdf, cdf_bins):
    """ 
    Normalize saliency to make saliency values distributed according to a given CDF
    """

    smap = saliency_map.copy()
    shape = smap.shape
    smap = smap.flatten()
    smap = np.argsort(np.argsort(smap)).astype(float)
    smap /= 1.0*len(smap)

    inds = np.searchsorted(cdf, smap, side='right')
    smap = cdf_bins[inds]
    smap = smap.reshape(shape)
    smap = smap.reshape(shape)
    return smap

def get_imlist(path):
  """  
  Returns a list of filenames for all jpg images in a directory. """

  return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg')]

#def Saliency_Map(img):
if __name__ == '__main__':
    '''
    Human visual inspired multi-layer LNL model. In this model, the main component
    are:

    Nature Image --> VonKries Adaptation --> ATD  (Color processing phase)
    Wavelets Transform --> Contrast sensivity function (CSF) --> Divisive
    Normalization(DN)  --> Noise(Gaussian or Poisson)

    Evalute of model with TID2008 database.

    Redundancy redunction measure with Total Correlation(RBIG or Cortex module)
    This model derivated two version script： Matlab, Python. 

    In the future, I want to implemented all of these code on C++ or Java. 
    If our goal is simulate of primate brain, we need to implement all everything 
    to High performance Computer(HPC) with big framework architecture(C/C++/Java).

    Input:
    -------
    Img: ndarray, float

    Output:
    --------
    Each layer response, float
    '''
    wavelets_types='DWT'
    threshold=False
    adaptation_gain_control=False
    statis_wavlets=True
    saturation =False
    dim = (256, 256)
    g_sa = 0.5
    epsilon_sa = 0.1
    k_sa = 1
    deltat_sa = 1e-5
    g_sa = np.array(g_sa).astype('float32')
    epsilon_sa = np.array(epsilon_sa).astype('float32')
    ############################################################################
    #Load image and preprocess
    ############################################################################  
    image=cv2.imread('/home/qiang/QiangLi/Python_Utils_Functional/BioMulti-L-NL-Model/imgs/tower.jpeg', cv2.IMREAD_UNCHANGED)
    print('Original Dimensions : ', image.shape)
    im= cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    print('Resized Dimensions : ',im.shape)
    im=im/255
    b,g,r = cv2.split(im)      
    rgb_img = cv2.merge([r,g,b])     
    
    plt.figure(figsize=(4,5))
    plt.imshow(rgb_img)
    plt.axis('off')
    plt.title("Orignal")
    plt.tight_layout()
    plt.show()

    print('########################')
    print('Step 1')
    print('Load img test passed!!!')
    print('########################')
    #---------------------------------------------------------------
    # Gamma correction with monit and eye nonlinearlity
    #---------------------------------------------------------------
    im_gamma=exposure.adjust_gamma(im, gamma=0.6)
    b,g,r = cv2.split(im_gamma)      
    img_gamma_rgb = cv2.merge([r,g,b])     
    
    plt.figure(figsize=(4,5))
    plt.imshow(img_gamma_rgb)
    plt.axis("off")
    plt.tight_layout()
    plt.title("GammaCorrection")
    plt.show()

    print('########################')
    print('Step 2')
    print('Gammma correction test passed!!!')
    print('########################')
    #---------------------------------------------------------------
    # VonKries adaptation
    #---------------------------------------------------------------
    if adaptation_gain_control==True:
        M = np.asarray([[.40024,0.7076,-.08081], [-.2263,1.16532,0.0457],[0,0,.91822]])
        int_M = inv(M)
        XYZ2BGR = np.asarray([[3.240479,-1.53715,-0.498535], [-0.969256,1.875991,0.041556],[0.055648,-0.204043,1.057311]])
        BGR2XYZ = inv(XYZ2BGR)

        L2 = 0.6
        M2 = 0.6
        S2 = 0.6

        width = im_gamma.shape[0]
        height = im_gamma.shape[1]
        img_LMS = np.zeros(im_gamma.shape)
        img_XYZ = np.zeros(im_gamma.shape)
        img_XYZ_corrected = np.zeros(im_gamma.shape)
        img_corrected = np.zeros(im_gamma.shape)

        for x in trange(width, desc='Running'):
            for y in trange(height, desc='Running'):
                img_XYZ[x,y,:] = np.matmul(BGR2XYZ,im_gamma[x,y,:]/255)
                
        for x in trange(width, desc='Running'):
            for y in trange(height, desc='Running'):
                img_LMS[x,y,:] = np.matmul(M,img_XYZ[x,y,:])

        L_max = img_LMS[:,:,0].max()
        M_max = img_LMS[:,:,1].max()
        S_max = img_LMS[:,:,2].max()

        img_LMS[:,:,0] = img_LMS[:,:,0]/L_max
        img_LMS[:,:,1] = img_LMS[:,:,1]/M_max
        img_LMS[:,:,2] = img_LMS[:,:,2]/S_max

        img_LMS[:,:,0] = img_LMS[:,:,0]*L2
        img_LMS[:,:,1] = img_LMS[:,:,1]*M2
        img_LMS[:,:,2] = img_LMS[:,:,2]*S2

        for x in trange(width, desc='Running'):
            for y in trange(height, desc='Running'):
                img_XYZ_corrected[x,y,:] = np.matmul(int_M,img_LMS[x,y,:])
        
        for x in trange(width, desc='Running'):
            for y in trange(height, desc='Running'):
                img_corrected[x,y,:] = np.matmul(XYZ2BGR,img_XYZ_corrected[x,y,:])

        results_RGB = np.zeros(img_XYZ.shape)
        results_RGB[:,:,0] = img_corrected[:,:,2]
        results_RGB[:,:,1] = img_corrected[:,:,1]
        results_RGB[:,:,2] = img_corrected[:,:,0]
        VonKries=(255*np.clip(results_RGB,0,1)).astype('uint8') 

        plt.figure(figsize=(4,5))
        plt.imshow((VonKries),vmin=results_RGB.min(),vmax=results_RGB.max())
        plt.axis("off")
        plt.title("VonKries")
        plt.tight_layout()
        plt.show()

        print('########################')
        print('Step 3')
        print('VonKries test passed!!!')
        print('########################')
    
    #----------------------------------------------------------------
    # Chromatic adaptation with different methods
    #----------------------------------------------------------------
    else:
        print(sorted(colour.CHROMATIC_ADAPTATION_TRANSFORMS))
        # Test view condition
        XYZ_w = np.array([234, 240, 220])
        # Reference view condition
        XYZ_wr = np.array([224, 187, 70]) 
        method = 'Von Kries'
        VonKries=colour.adaptation.chromatic_adaptation_VonKries(rgb2xyz(im_gamma), XYZ_w, XYZ_wr, method)
        try:
            print(VonKries.shape)
        except ValueError:
            print('Dimentional error!') 

        fig=plt.figure(figsize=(4,5))
        fig.add_subplot(111)
        plt.imshow(VonKries)
        plt.title('VonKriesAdaptation')
        plt.tight_layout()
        plt.axis('off')
        plt.show()
    #---------------------------------------------------------------
    # Colorsapce conversation- destation ATD
    # Visualization channel with fake color
    #---------------------------------------------------------------
    plt.figure(figsize=(10,8))
    plt.subplots_adjust(wspace=0, hspace=0)

    xyz_=rgb2xyz(VonKries)
    IOU=xyz2atd(xyz_)

    plt.subplot(1, 3, 1)
    plt.imshow(IOU[:,:,0],cmap='gray')
    plt.axis('off')
    plt.title("A")

    plt.subplot(1, 3, 2)
    plt.imshow(IOU[:,:,1],cmap='gray')
    plt.axis('off')
    plt.title("T")

    plt.subplot(1, 3, 3)
    plt.imshow(IOU[:,:,2],cmap='gray')
    plt.axis('off')
    plt.title("D")
    plt.tight_layout()
    plt.show()
    
    print('########################')
    print('Step 4')
    print('ATD test passed!!!')
    print('########################')
    #---------------------------------------------------------------
    # Weber law application
    #---------------------------------------------------------------
    waber_img=waber(IOU[:,:,0], lambdaValue=0.6)
    plt.figure(figsize=(4,5))
    plt.imshow(waber_img, cmap='gray')
    plt.axis('off')
    plt.title('brightness')
    plt.tight_layout()
    plt.show()
    
    print('########################')
    print('Step 5')
    print('Weber law test passed!!!')
    print('########################')
      
  # 5) -- DWT
  ############################################################################
  # Build Pyramid with DWT
  ############################################################################
    max_lev = 3
    level=max_lev   
    shape=waber_img.shape
    c = pywt.wavedec2(waber_img, 'db2', mode='periodization', level=level)
    c[0] /= np.abs(c[0]).max()
    for detail_level in trange(level, desc='Running'):
        c[detail_level + 1] = [d/np.abs(d).max() for d in c[detail_level + 1]]

    arr, slices = pywt.coeffs_to_array(c)
    plt.figure(figsize=(8,6))
    plt.imshow(arr, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    print('######################################')
    print('Step 6')
    print('Build pyramid with DWT test passed!!!')
    print('#####################################')

    ##################################################
    # Wavelet energy map
    ##################################################
    Energ_levels=[]
    for i in tqdm(range(1, max_lev+1), desc='Running'):
        for j in tqdm(range(max_lev), desc='Running'):
            Energ_level=np.abs(c[i][j])**2
            Energ_levels.append(Energ_level)
    #---------------------------------------------------
    #Energy map with grid show
    #---------------------------------------------------
    Init_en=np.abs(c[0])**2
    Eng_grid=[Init_en, Energ_levels[:3], Energ_levels[3:6], Energ_levels[6:9]]

    Eng_grid[0] /= np.abs(Eng_grid[0]).max()
    for detail_level in trange(level, desc='Running'):
        Eng_grid[detail_level + 1] = [d/np.abs(d).max() for d in Eng_grid[detail_level + 1]]
    arr_, slices_ = pywt.coeffs_to_array(Eng_grid)
    plt.figure(figsize=(8,6))
    plt.imshow(arr_, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    #---------------------------------------------------
    #Energy map with flatten map
    #---------------------------------------------------
    fin=Energ_levels[0]+Energ_levels[1]+Energ_levels[2]
    fina=Energ_levels[3]+Energ_levels[4]+Energ_levels[5]
    finb=Energ_levels[6]+Energ_levels[7]+Energ_levels[8]

    plt.figure(figsize=(12,10))
    plt.subplots_adjust(wspace=0.1)
    ener_q=[fin, fina, finb]
    for i in tqdm(range(len(ener_q)), desc='Running'):
        plt.subplot(1,3,i+1)
        plt.imshow(ener_q[i], cmap='gray')
        plt.axis('off')
    plt.show()
    #---------------------------------------------------
    #Saliency Map with UpSample Scale Together
    #---------------------------------------------------
    print(fin.shape)
    print(fina.shape)
    print(finb.shape)
    fin_upsample = scipy.ndimage.zoom(fin, 4, order=0)
    fina_upsample = scipy.ndimage.zoom(fina, 2, order=0)
    print(fin_upsample.shape)
    print(fina_upsample.shape)  
    Saliency = fin_upsample + fina_upsample + finb 
    plt.figure(figsize=(12,10))
    plt.imshow(Saliency, cmap='gray')
    plt.axis('off')
    plt.title('Saliency Map')
    plt.show()