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

#%tensorflow_version 1x
import tensorflow as tf
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
#                          Visual Computing
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

  # The nonlinearlity convert between human visual and display screen 

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
  'Hard threshold for Ortho-wavelets'
  return u*(np.abs(u)>t)
  
def thresh_soft(u,t):
  'Soft threshold for Ortho-wavelets'
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
def saturation_f(x,g,xm,epsilon,sizeT):
  '''
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

  '''
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


def SimpleVisualModel_Beta(img):
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
  ############################################################################
  # Test with real image
  # Copyright 2020 QiangLi
  ############################################################################
  # Main section 
  ############################################################################
  noise_types=['Gaussian', 'Possion']
  noise_type=noise_types[0]
  expont = 2.2
  ############################################################################
  #Load image and preprocess
  ############################################################################  
  print('\n Original Dimensions:', img.shape)
  lumin_img = img
  ############################################################################
  #Wavelet transform
  ############################################################################
  filt = 'sp3_filters' 
  pyr = pt.pyramids.SteerablePyramidSpace(lumin_img, height=3, order=3)
  imgCoeffs = []
  imgCoeffs_Iwant = []
  for j in trange(pyr.num_scales, desc='Running'):
    for s in trange(pyr.num_scales+1, desc='Running'):
      band = pyr.pyr_coeffs[j,s]
      imgCoeffs.append(band)
      band_s = np.asarray(band.reshape(band.shape[0]*band.shape[1]))
      imgCoeffs_Iwant.append(band_s)

  ############################################################################
  # CSF -  operate in the Fourier space then convert back to spatial domain
  #     -  all wavelets domain 
  ############################################################################
  CSF_wavelets_vis_sp=[]
  Dis_CSF_wavelets_vis_sp = []
  CSF_wavelets_sp=[]
  CSF_wavelet_m_sp=[]
  CSF_wavelet_h_sp=[]
  CSF_wavelet_s_sp=[]
 
  for t_sp in tqdm(range(len(imgCoeffs))[0:4], desc='Running'):  
    res_M_sp=make_CSF(x=128, nfreq=32)
    Img_CSF_M_sp = np.double (np.real(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(imgCoeffs[t_sp].real))* res_M_sp))))
    CSF_wavelet_m_sp.append(Img_CSF_M_sp)
    CSF_wavelets_vis_sp.append(Img_CSF_M_sp)
    Img_CSF_M_sp = np.asarray(Img_CSF_M_sp.reshape(Img_CSF_M_sp.shape[0]*Img_CSF_M_sp.shape[1]))
    Dis_CSF_wavelets_vis_sp.append(Img_CSF_M_sp)
  for y_sp in tqdm(range(len(imgCoeffs))[4:8], desc='Running'):  
    res_H_sp=make_CSF(x=64, nfreq=32)
    Img_CSF_H_sp = np.double (np.real(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(imgCoeffs[y_sp].real))* res_H_sp))))
    CSF_wavelet_h_sp.append(Img_CSF_H_sp)
    CSF_wavelets_vis_sp.append(Img_CSF_H_sp)
    Img_CSF_H_sp = np.asarray(Img_CSF_H_sp.reshape(Img_CSF_H_sp.shape[0]*Img_CSF_H_sp.shape[1]))
    Dis_CSF_wavelets_vis_sp.append(Img_CSF_H_sp)
  for e_sp in tqdm(range(len(imgCoeffs))[8:12], desc='Running'):  
    res_S_sp=make_CSF(x=32, nfreq=32)
    Img_CSF_S_sp = np.double (np.real(np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(imgCoeffs[e_sp].real))* res_S_sp))))
    CSF_wavelet_s_sp.append(Img_CSF_S_sp)
    CSF_wavelets_vis_sp.append(Img_CSF_S_sp)
    Img_CSF_S_sp = np.asarray(Img_CSF_S_sp.reshape(Img_CSF_S_sp.shape[0]*Img_CSF_S_sp.shape[1]))
    Dis_CSF_wavelets_vis_sp.append(Img_CSF_S_sp)
  
  CSF_wavelets_sp=[CSF_wavelet_m_sp[0], CSF_wavelet_m_sp, CSF_wavelet_h_sp, CSF_wavelet_s_sp, CSF_wavelet_s_sp[0]]

  if len(CSF_wavelets_sp)==5:
    print('CSF filter was right!!')
  else:
    print('CSF filter test wrong, go back check!!')

  ############################################################################
  # Divisive Normalization
  ############################################################################
  DN_band_sp=[]
  Dis_DN_band_sp=[]
   
  for k in tqdm(range(0, 4), desc='Running'):
    reconstruction_num_level_m_sp = np.reshape(CSF_wavelets_vis_sp[k], (128, 128, 1))
    DN_m_sp = DivisiveNormalization(torch.Tensor([reconstruction_num_level_m_sp.transpose((2,0,1))]),radius=9) 
    ret_m_sp = DN_m_sp[0].numpy().transpose((1,2,0)) 
    scaled_ret_m_sp = (ret_m_sp - ret_m_sp.min())/(ret_m_sp.max() - ret_m_sp.min()) 
    scaled_ret_m_sp = np.reshape(scaled_ret_m_sp, (128, 128))
    DN_band_sp.append(scaled_ret_m_sp)
    scaled_ret_m_sp = np.asarray(scaled_ret_m_sp.reshape(scaled_ret_m_sp.shape[0]*scaled_ret_m_sp.shape[1]))
    Dis_DN_band_sp.append(scaled_ret_m_sp)
    
  for p in tqdm(range(4, 8), desc='Running'):
    reconstruction_num_level_h_sp = np.reshape(CSF_wavelets_vis_sp[p], (64, 64, 1))
    DN_h_sp = DivisiveNormalization(torch.Tensor([reconstruction_num_level_h_sp.transpose((2,0,1))]),radius=9) 
    ret_h_sp = DN_h_sp[0].numpy().transpose((1,2,0)) 
    scaled_ret_h_sp = (ret_h_sp - ret_h_sp.min())/(ret_h_sp.max() - ret_h_sp.min()) 
    scaled_ret_h_sp = np.reshape(scaled_ret_h_sp, (64, 64))
    DN_band_sp.append(scaled_ret_h_sp)
    scaled_ret_h_sp = np.asarray(scaled_ret_h_sp.reshape(scaled_ret_h_sp.shape[0]*scaled_ret_h_sp.shape[1]))
    Dis_DN_band_sp.append(scaled_ret_h_sp)
    
  for q in tqdm(range(8, 12), desc='Running'):
    reconstruction_num_level_s_sp = np.reshape(CSF_wavelets_vis_sp[q], (32, 32, 1))
    DN_s_sp = DivisiveNormalization(torch.Tensor([reconstruction_num_level_s_sp.transpose((2,0,1))]),radius=9) 
    ret_s_sp = DN_s_sp[0].numpy().transpose((1,2,0)) 
    scaled_ret_s_sp = (ret_s_sp - ret_s_sp.min())/(ret_s_sp.max() - ret_s_sp.min()) 
    scaled_ret_s_sp = np.reshape(scaled_ret_s_sp, (32, 32))
    DN_band_sp.append(scaled_ret_s_sp)
    scaled_ret_s_sp = np.asarray(scaled_ret_s_sp.reshape(scaled_ret_s_sp.shape[0]*scaled_ret_s_sp.shape[1]))
    Dis_DN_band_sp.append(scaled_ret_s_sp)
    
  try:
    print(len(DN_band_sp))
    if len(DN_band_sp)==12:
      print('DN model works!')
  except ValueError:
    print('DN model errors')

  ############################################################################
  # Neural noise - 1) Gaussian  2) Possion 
  ############################################################################
  #Gaussian
  f_gaussian_sp=[]
  sigma=.1

  if noise_type=='Gaussian':
    for n in tqdm(range(0, 4), desc='Running'):
      f_gaussian_sp_n = DN_band_sp[n]*np.random.standard_normal(DN_band_sp[n].shape)
      f_gaussian_sp_n = np.asarray(f_gaussian_sp_n.reshape(f_gaussian_sp_n.shape[0]*f_gaussian_sp_n.shape[1]))
      f_gaussian_sp.append(f_gaussian_sp_n)
    for y in tqdm(range(4, 8), desc='Running'):
      f_gaussian_sp_y = DN_band_sp[y]*np.random.standard_normal(DN_band_sp[y].shape)
      f_gaussian_sp_y = np.asarray(f_gaussian_sp_y.reshape(f_gaussian_sp_y.shape[0]*f_gaussian_sp_y.shape[1]))
      f_gaussian_sp.append(f_gaussian_sp_y)
    for p in tqdm(range(8, 12), desc='Running'):
      f_gaussian_sp_p = DN_band_sp[p]*np.random.standard_normal(DN_band_sp[p].shape)
      f_gaussian_sp_p = np.asarray(f_gaussian_sp_p.reshape(f_gaussian_sp_p.shape[0]*f_gaussian_sp_p.shape[1]))
      f_gaussian_sp.append(f_gaussian_sp_p)
  
  return imgCoeffs_Iwant, Dis_CSF_wavelets_vis_sp, Dis_DN_band_sp, f_gaussian_sp 
  
  #Possion
  f_possion_sp=[]
  lam=25.5
  if noise_type=='Possion':
    for n in tqdm(range(0, 4), desc='Running'):  
      f_possion_sp_n = np.random.poisson(DN_band_sp[n]*lam, DN_band_sp[n].shape)
      f_possion_sp_n = np.asarray(f_possion_sp_n.reshape(f_possion_sp_n.shape[0]*f_possion_sp_n.shape[1]))
      f_possion_sp.append(f_possion_sp_n)
    for y in tqdm(range(4, 8), desc='Running'):  
      f_possion_sp_y = np.random.poisson(DN_band_sp[y]*lam, DN_band_sp[y].shape)
      f_possion_sp_y = np.asarray(f_possion_sp_y.reshape(f_possion_sp_y.shape[0]*f_possion_sp_y.shape[1]))
      f_possion_sp.append(f_possion_sp_y)
    for p in tqdm(range(8, 12), desc='Running'):  
      f_possion_sp_p = np.random.poisson(DN_band_sp[p]*lam, DN_band_sp[p].shape)
      f_possion_sp_p = np.asarray(f_possion_sp_p.reshape(f_possion_sp_p.shape[0]*f_possion_sp_p.shape[1]))
      f_possion_sp.append(f_possion_sp_p)
  
  return imgCoeffs_Iwant, Dis_CSF_wavelets_vis_sp, Dis_DN_band_sp, f_possion_sp    

def SFF(Ir, Id):
  '''
  Euclidean distance calculate equal perceptual distance
  The Simple visual model implemented before and here we
  use TID2008 database to evaluate model performance.

  Input:
  ------
  Ir: Reference Image,  ndarray, float(patches)
  Id: Distorcted Image, ndarray, float(patches)

  Output:
  -------
  perceptual Distance, list, float
  '''
  param_N = 128
  
  patches_r_r = im2col_sliding_strided(Ir, [param_N, param_N], stepsize=64)
  patches_d_d = im2col_sliding_strided(Id, [param_N, param_N], stepsize=64)

  for p in range(len(patches_r_r[0,:])):
    patches_r = np.reshape(patches_r_r[:,p],[param_N, param_N])
    patches_d = np.reshape(patches_d_d[:,p],[param_N, param_N])

    w, wf, ws, wsn  = SimpleVisualModel_Beta(patches_r) 
    wd, wfd, wsd, wsnd = SimpleVisualModel_Beta(patches_d)

    global D
    global DF 
    global DFS 
    global DFSN

    D_up = []
    DF_up = []
    DFS_up = []
    DFSN_up = []
    
    if p==0:
      D = np.zeros((len(w),len(patches_r[0,:])))
      DF = np.zeros((len(w),len(patches_r[0,:])))
      DFS = np.zeros((len(w),len(patches_r[0,:])))
      DFSN = np.zeros((len(w),len(patches_r[0,:])))
      
      D = map(sub, w, wd)
      for x1 in D:
          D_up.append(x1)
      DF = map(sub, wf, wfd)
      for x2 in DF:
          DF_up.append(x2)
      DFS = map(sub, ws, wsd)
      for x3 in DFS:
          DFS_up.append(x3)
      DFSN = map(sub, wsn, wsnd)
      for x4 in DFSN:
          DFSN_up.append(x4)
    else:
      D = map(sub, w, wd)
      for x5 in D:
          D_up.append(x5)
      DF = map(sub, wf, wfd)
      for x6 in DF:
          DF_up.append(x6)
      DFS = map(sub, ws, wsd)
      for x7 in DFS:
          DFS_up.append(x7)
      DFSN = map(sub, wsn, wsnd)
      for x8 in DFSN:
          DFSN_up.append(x8)
   
    sum_exponent = 2
    
    #dor =  np.sqrt(np.sum(np.power(np.abs(Ir[:] - Id[:]), 2)))
    dr = np.sqrt(np.sum(np.power(np.abs(patches_r[:] - patches_d[:]), 2)))
    dw = np.power(np.sum(np.concatenate(np.power(np.abs(D_up[:]), sum_exponent))), (1/sum_exponent))
    dwf = np.power(np.sum(np.concatenate(np.power(np.abs(DF_up[:]), sum_exponent))), (1/sum_exponent))
    dwfs = np.power(np.sum(np.concatenate(np.power(np.abs(DFS_up[:]), sum_exponent))), (1/sum_exponent))
    dwfsn = np.power(np.sum(np.concatenate(np.power(np.abs(DFSN_up[:]), sum_exponent))), (1/sum_exponent))

  return dr, dw, dwf, dwfs, dwfsn


###############################################################################################################
if __name__ == '__main__':

  mos = io.loadmat('/home/qiang/QiangLi/Python_Utils_Functional/FirstVersion-BioMulti-L-NL-Model-ongoing/TID2008/TID2008.mat')
  print(len((mos['tid_MOS'])))
  tid_MOS = mos['tid_MOS']


  ScoreSingle = np.zeros([68, 1])
  iPoint = 0

  expo = 2.2 
  expo = 1  
  param_n = 256

  indices = []

  drq = []
  dwq = []
  dwfq = []
  dwfsq = []
  dwfsnq = []
  
  start = time.time()
  for iRef in range(2,6):
    imNameRef = str("{:02d}".format(iRef))
    print(imNameRef)
    Ir = imread(os.path.join('/home/qiang/QiangLi/Python_Utils_Functional/FirstVersion-BioMulti-L-NL-Model-ongoing/TID2008/reference_images/I' + imNameRef + '.BMP'))
    Ir = resize(Ir, (param_n, param_n))
    Iro = rgb2gray(np.double(Ir)/255)
    print(Iro.shape)
    Ir = np.power(Iro, expo)
    for iDis in range(1,18):
      imNameDis = os.path.join('_' + str('{:02d}'.format(iDis)))
      for iLevel in range(1, 5):
        index = (iRef-1)*68 + (iDis-1)*4 + iLevel
        indices.append([iRef, iDis, iLevel, index])
        print('The index image now is {}'.format(indices))
        Id = imread(os.path.join('/home/qiang/QiangLi/Python_Utils_Functional/FirstVersion-BioMulti-L-NL-Model-ongoing/TID2008/distorted_images/I' + imNameRef + imNameDis + '_' + str(iLevel) + '.bmp'))
        Id = resize(Id, (param_n, param_n))
        Ido=rgb2gray(np.double(Id)/255)
        print(Ido.shape)
        Id=np.power(Ido, expo)
        #dr[iPoint], dw[iPoint], dwf[iPoint], dwfs[iPoint], dwfsn[iPoint] = SFF(Ir,Id)
        dr, dw, dwf, dwfs, dwfsn = SFF(Ir,Id)
        drq.append(dr)
        dwq.append(dw)
        dwfq.append(dwf)
        dwfsq.append(dwfs)
        dwfsnq.append(dwfsn)
  
  print('Time Taken: {:.2f}'.format(time.time() - start))      
  
  np.save('/home/qiang/QiangLi/Python_Utils_Functional/BioMulti-L-NL-Model/database/indices', indices)
  np.save('/home/qiang/QiangLi/Python_Utils_Functional/BioMulti-L-NL-Model/database/drq', drq)
  np.save('/home/qiang/QiangLi/Python_Utils_Functional/BioMulti-L-NL-Model/database/dwq', dwq)
  np.save('/home/qiang/QiangLi/Python_Utils_Functional/BioMulti-L-NL-Model/database/dwfq', dwfq)
  np.save('/home/qiang/QiangLi/Python_Utils_Functional/BioMulti-L-NL-Model/database/dwfsq', dwfsq)
  np.save('/home/qiang/QiangLi/Python_Utils_Functional/BioMulti-L-NL-Model/database/dwfsnq', dwfsnq)
  
'''
  ####################################################################################################
  # The result visualization with plot_corr_mos.py
  ####################################################################################################
  indi = indices[0][0] != 0                            
  SB = tid_MOS[indices[indi:]].T

  ##################################################
  #Visualization 
  ##################################################
  plt.figure(figsize=(15,12), dpi=144)
  plt.subplots_adjust(wspace=0.3, hspace=0)
  plt.margins(0,0)

  OB = drq[indi]                
  metric_1 = np.corrcoef(SB, OB) 
  plt.subplot(141)
  plt.scatter(SB, OB, 'b.')
  plt.axis('equal')
  plt.title('r = {}'.format(metric_1))

  OB = dwq[indi]                
  metric_2 = np.corrcoef(SB, OB)  
  plt.subplot(142)
  plt.scatter(SB, OB, 'b.')
  plt.axis('equal')
  plt.title('r = {}'.format(metric_2))

  OB = dwfq[indi]  
  metric_3 = np.corrcoef(SB, OB)  
  plt.subplot(143)
  plt.scatter(SB, OB, 'b.')
  plt.axis('equal')
  plt.title('r = {}'.format(metric_3))

  OB = dwfsq[indi]  
  metric_4 = np.corrcoef(SB, OB)  
  plt.subplot(144)
  plt.scatter(SB, OB, 'b.')
  plt.axis('equal')
  plt.title('r = {}'.format(metric_4))

  OB = dwfsnq[indi]  
  metric_5 = np.corrcoef(SB, OB)  
  plt.subplot(145)
  plt.scatter(SB, OB, 'b.')
  plt.axis('equal')
  plt.title('r = {}'.format(metric_5))

  plt.tight_layout()

plt.show()
'''