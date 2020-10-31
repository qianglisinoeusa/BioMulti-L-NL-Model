
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
'''
sys.path.insert(0,'/home/qiang/QiangLi/Python_Utils_Functional/FixaTons-master/')
import FixaTons
# Put the dataset into the same fold with code
dataset = 'MIT1003'
FixaTons.info.stimuli('MIT1003')
DATASET_NAME = 'MIT1003'

for dataset in FixaTons.info.datasets():
    for image in FixaTons.info.stimuli('MIT1003'):
        FixaTons.show.map(dataset, image, plotMaxDim=1500)
        for subject in FixaTons.info.subjects(dataset, image):
            FixaTons.show.scanpath(dataset, image, subject, animation=True, plotMaxDim=1000, wait_time=10) 
'''

def get_imlist(path):
  """  
  Returns a list of filenames for all jpg images in a directory. """

  return [os.path.join(path,f) for f in os.listdir(path) if f.endswith(('.jpeg', '.png'))]


def normalize_saliency_map(saliency_map, cdf, cdf_bins):
    """ Normalize saliency to make saliency values distributed according to a given CDF
    """

    smap = saliency_map.copy()
    shape = smap.shape
    smap = smap.flatten()
    smap = np.argsort(np.argsort(smap)).astype(float)
    smap /= 1.0 * len(smap)

    inds = np.searchsorted(cdf, smap, side='right')
    smap = cdf_bins[inds]
    smap = smap.reshape(shape)
    smap = smap.reshape(shape)
    return smap


def convert_saliency_map_to_density(saliency_map, minimum_value=0.0):
    if saliency_map.min() < 0:
        saliency_map = saliency_map - saliency_map.min()
    saliency_map = saliency_map + minimum_value

    saliency_map_sum = saliency_map.sum()
    if saliency_map_sum:
        saliency_map = saliency_map / saliency_map_sum
    else:
        saliency_map[:] = 1.0
        saliency_map /= saliency_map.sum()

    return saliency_map

def probabilistic_image_based_kl_divergence(logp1, logp2, log_regularization=0, quotient_regularization=0):
    if log_regularization or quotient_regularization:
        return (np.exp(logp2) * np.log(log_regularization + np.exp(logp2) / (np.exp(logp1) + quotient_regularization))).sum()
    else:
        return (np.exp(logp2) * (logp2 - logp1)).sum()

def image_based_kl_divergence(saliency_map_1, saliency_map_2, minimum_value=1e-20, log_regularization=0, quotient_regularization=0):
    """ KLDiv. Function is not symmetric. saliency_map_2 is treated as empirical saliency map. """
    log_density_1 = np.log(convert_saliency_map_to_density(saliency_map_1, minimum_value=minimum_value))
    log_density_2 = np.log(convert_saliency_map_to_density(saliency_map_2, minimum_value=minimum_value))

    return probabilistic_image_based_kl_divergence(log_density_1, log_density_2, log_regularization=log_regularization, quotient_regularization=quotient_regularization)


#Metric 1
def MIT_KLDiv(saliency_map_1, saliency_map_2):
    """ compute image-based KL divergence with same hyperparameters as in Tuebingen/MIT Saliency Benchmark """
    return image_based_kl_divergence(
        saliency_map_1,
        saliency_map_2,
        minimum_value=0,
        log_regularization=2.2204e-16,
        quotient_regularization=2.2204e-16
    )

def kldiv(s_map,gt):
	s_map = s_map/(np.sum(s_map)*1.0)
	gt = gt/(np.sum(gt)*1.0)
	eps = 2.2204e-16
	return np.sum(gt * np.log(eps + gt/(s_map + eps)))


#Metric 2
def SIM(saliency_map_1, saliency_map_2):
    """ Compute similiarity metric. """
    density_1 = convert_saliency_map_to_density(saliency_map_1, minimum_value=0)
    density_2 = convert_saliency_map_to_density(saliency_map_2, minimum_value=0)

    return np.min([density_1, density_2], axis=0).sum()

#Metric 3
def CC(saliency_map_1, saliency_map_2):
    def normalize(saliency_map):
        saliency_map -= saliency_map.mean()
        std = saliency_map.std()

        if std:
            saliency_map /= std

        return saliency_map, std == 0

    smap1, constant1 = normalize(saliency_map_1.copy())
    smap2, constant2 = normalize(saliency_map_2.copy())

    if constant1 and not constant2:
        return 0.0
    else:
        return np.corrcoef(smap1.flatten(), smap2.flatten())[0, 1]

#Metric 4
def NSS(saliency_map, xs, ys):
    xs = np.asarray(xs, dtype=np.int)
    ys = np.asarray(ys, dtype=np.int)

    mean = saliency_map.mean()
    std = saliency_map.std()

    value = saliency_map[ys, xs].copy()
    value -= mean

    if std:
        value /= std

    return value

#Visualization
def plot_information_gain(information_gain, ax=None, color_range = None, image=None, frame=False,
                          thickness = 1.0, zoom_factor=1.0, threshold=0.05, rel_levels=None,
                          alpha=0.5, color_offset = 0.25, plot_color_bar=True):
    """
    Create pixel space information gain plots as in the paper.
    Parameters:
    -----------
    information gain: the information gain to plot.
    ax: the matplotlib axes object to use. If none, use current axes.
    color_range: Full range of colorbar
    """
    if ax is None:
        ax = plt.gca()
    ig = information_gain

    if zoom_factor != 1.0:
        ig = zoom(ig, zoom_factor, order=0)

    if color_range is None:
        color_range = (ig.min(), ig.max())
    if not isinstance(color_range, (tuple, list)):
        color_range = (-color_range, color_range)

    color_total_max = max(np.abs(color_range[0]), np.abs(color_range[1]))

    if image is not None:
        if image.ndim == 3:
            image = image.sum(axis=-1)
        ax.imshow(image, alpha=0.3)

    if rel_levels is None:
        rel_levels = [0.1, 0.4, 0.7]

    # from https://stackoverflow.com/questions/8580631/transparent-colormap
    cm = plt.cm.get_cmap('hsv') #RdBu
    cm._init()
    alphas = (np.abs(np.linspace(-1.0, 1.0, cm.N)))
    alphas = np.ones_like(alphas)*alpha
    cm._lut[:-3, -1] = alphas

    levels = []
    colors = []

    min_val = np.abs(ig.min())
    max_val = np.abs(ig.max())

    total_max = max(min_val, max_val)

    def get_color(val):
        # value relative -1 .. 1
        rel_val = val / color_total_max
        # shift around 0
        rel_val = (rel_val + np.sign(rel_val) * color_offset) / (1+color_offset)
        # transform to 0 .. 1
        rel_val = (0.5 + rel_val / 2)
        return cm(rel_val)

    if min_val / total_max > threshold:
        for l in [1.0]+rel_levels[::-1]:
            val = -l*min_val
            levels.append(val)

            colors.append(get_color(val))
    else:
        levels.append(-total_max)
        colors.append('white')

    # We want to use the color from the value nearer to zero
    colors = colors[1:]
    colors.append((1.0, 1.0, 1.0, 0.0))

    if max_val / total_max > threshold:
        for l in rel_levels+[1.0]:
            val = l*max_val
            levels.append(val)

            colors.append(get_color(val))
    else:
        levels.append(total_max)

    #print rel_vals
    ax.contourf(ig, levels=levels,
                colors=colors,
                vmin=-color_total_max, vmax=color_total_max
                )
    ax.contour(ig, levels=levels,
               # colors=colors,
               #          vmin=-color_range, vmax=color_range
               colors = 'gray',
               linestyles='solid',
               linewidths=0.6*thickness
               )

    if plot_color_bar:
        ## Draw color range bar
        h = 100
        w = 10
        t = np.empty((100, 10, 4))
        for y in range(h):
            for x in range(w):
                val = (y/h) * (color_range[1] - color_range[0]) + color_range[0]
                color = np.asarray(get_color(val))
                if not -min_val <= val <= max_val:
                    color[-1] *= 0.4
                else:
                    color[-1] = 1
                t[y, x, :] = color

        ax.imshow(t, extent=(0.95*ig.shape[1], 0.98*ig.shape[1],
                             0.1*ig.shape[0], 0.9*ig.shape[0]))

    ax.set_xlim(0, ig.shape[1])
    ax.set_ylim(ig.shape[0], 0)

    if frame:
        # Just a frame
        ax.set_xticks([])
        ax.set_yticks([])
        [i.set_linewidth(i.get_linewidth()*thickness) for i in ax.spines.itervalues()]
    else:
        ax.set_axis_off()


def normalize_log_density(log_density):
    """ convertes a log density into a map of the cummulative distribution function.
    """
    density = np.exp(log_density)
    flat_density = density.flatten()
    inds = flat_density.argsort()[::-1]
    sorted_density = flat_density[inds]
    cummulative = np.cumsum(sorted_density)
    unsorted_cummulative = cummulative[np.argsort(inds)]
    return unsorted_cummulative.reshape(log_density.shape)

def visualize_distribution(log_densities, ax = None):
    if ax is None:
        ax = plt.gca()
    t = normalize_log_density(log_densities)
    img = ax.imshow(t, cmap=plt.cm.viridis)
    levels = levels=[0, 0.25, 0.5, 0.75, 1.0]
    cs = ax.contour(t, levels=levels, colors='black')
    #plt.clabel(cs)

    return img, cs

def adjust_image_size(img1, img2, downscale_only = False):
    '''
    Checks if the images have the same size. If not, it rescales the first image to the size of the second image. 
    If downscale_only is set to true, the bigger one is scaled to the size of the smaller image, regardless of the ordering.
    '''
    # make the images the same size
    if downscale_only:
        # scale the bigger map to the smaller size
        if (np.size(img1,0) > np.size(img2,0) or np.size(img1,1) > np.size(img2,1)):
            img1 = resize(img1, np.shape(img2), mode='constant', anti_aliasing=True)
            
        elif (np.size(img1,0) < np.size(img2,0) or np.size(img1,1) < np.size(img2,1)):
            img2 = resize(img2, np.shape(img1), mode='constant', anti_aliasing=True)
    
    else:
        # scale both maps to the size of the first map
        if (np.size(img1, 0) != np.size(img2, 0) or np.size(img1, 1) != np.size(img2, 1)):
            img1 = resize(img1, np.shape(img2), mode='constant', anti_aliasing=True)
    
    return img1, img2 

def center_bias(func, mapsize):
    g = np.zeros(mapsize)
    for xi in range(0, mapsize[0]):
        for yi in range(0,mapsize[1]):
            x = xi-mapsize[0]/2
            y = yi-mapsize[1]/2
            g[xi, yi] = func(x, y)
    g = g / np.max(g)

    return g

def infogain(s_map,gt,baseline_map):
	gt = discretize_gt(gt)
	# assuming s_map and baseline_map are normalized
	eps = 2.2204e-16

	s_map = s_map/(np.sum(s_map)*1.0)
	baseline_map = baseline_map/(np.sum(baseline_map)*1.0)

	# for all places where gt=1, calculate info gain
	temp = []
	x,y = np.where(gt==1)
	for i in zip(x,y):
		temp.append(np.log2(eps + s_map[i[0],i[1]]) - np.log2(eps + baseline_map[i[0],i[1]]))

	return np.mean(temp)

def compute_information_gain(sal_map, fix_binary, baseline = []):
    '''
    Computes the information gain of the saliency map over the baseline. If no baseline is provided,
    the information gain over the center bias is computed.
    '''
    def gaussian2D(x, y, sigma):
        return (1.0/(2*math.pi*(sigma**2)))*math.exp(-(1.0/(2*(sigma**2)))*(x**2 + y**2))
    
    # adjust image size if it hasn't happened before
    sal_map, fix_binary = adjust_image_size(sal_map, fix_binary)
    # create center bias baseline if no baseline provided
    baseline = baseline if list(baseline) else center_bias(lambda x, y: gaussian2D(x, y, 50), np.shape(fix_binary))
    
    # bring the maps to probability distribution
    sal_map = sal_map - np.min(sal_map)
    sal_map = sal_map / np.sum(sal_map)
    baseline = baseline - np.min(baseline)
    baseline = baseline / np.sum(baseline)
    
    # compute information gain
    epsilon = np.finfo('float64').eps
    ig = np.sum(fix_binary * (np.log2(epsilon + sal_map) - np.log2(epsilon + baseline))) / np.sum(fix_binary)
    
    return ig


def normalize_map(s_map):
	norm_s_map = (s_map - np.min(s_map))/((np.max(s_map)-np.min(s_map))*1.0)
	return norm_s_map


def AUC_Judd(saliencyMap, fixationMap, jitter=True, toPlot=False):
    # saliencyMap is the saliency map
    # fixationMap is the human fixation map (binary matrix)
    # jitter=True will add tiny non-zero random constant to all map locations to ensure
    # 		ROC can be calculated robustly (to avoid uniform region)
    # if toPlot=True, displays ROC curve

    # If there are no fixations to predict, return NaN
    if not fixationMap.any():
        print('Error: no fixationMap')
        score = float('nan')
        return score

    # make the saliencyMap the size of the image of fixationMap

    if not np.shape(saliencyMap) == np.shape(fixationMap):
        saliencyMap = resize(saliencyMap, np.shape(fixationMap))

    # jitter saliency maps that come from saliency models that have a lot of zero values.
    # If the saliency map is made with a Gaussian then it does not need to be jittered as
    # the values are varied and there is not a large patch of the same value. In fact
    # jittering breaks the ordering in the small values!
    if jitter:
        # jitter the saliency map slightly to distrupt ties of the same numbers
        saliencyMap = saliencyMap + np.random.random(np.shape(saliencyMap)) / 10 ** 7

    # normalize saliency map
    saliencyMap = (saliencyMap - saliencyMap.min()) \
                  / (saliencyMap.max() - saliencyMap.min())

    if np.isnan(saliencyMap).all():
        print('NaN saliencyMap')
        score = float('nan')
        return score

    S = saliencyMap.flatten()
    F = fixationMap.flatten()

    Sth = S[F > 0]  # sal map values at fixation locations
    Nfixations = len(Sth)
    Npixels = len(S)

    allthreshes = sorted(Sth, reverse=True)  # sort sal map values, to sweep through values
    tp = np.zeros((Nfixations + 2))
    fp = np.zeros((Nfixations + 2))
    tp[0], tp[-1] = 0, 1
    fp[0], fp[-1] = 0, 1

    for i in range(Nfixations):
        thresh = allthreshes[i]
        aboveth = (S >= thresh).sum()  # total number of sal map values above threshold
        tp[i + 1] = float(i + 1) / Nfixations  # ratio sal map values at fixation locations
        # above threshold
        fp[i + 1] = float(aboveth - i) / (Npixels - Nfixations)  # ratio other sal map values
        # above threshold

    score = np.trapz(tp, x=fp)
    allthreshes = np.insert(allthreshes, 0, 0)
    allthreshes = np.append(allthreshes, 1)
    if toPlot:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 2, 1)
        ax.matshow(saliencyMap, cmap='gray')
        ax.set_axis_off()
        ax.set_title('SaliencyMap with fixations to be predicted')
        [y, x] = np.nonzero(fixationMap)
        s = np.shape(saliencyMap)
        plt.axis((-.5, s[1] - .5, s[0] - .5, -.5))
        plt.plot(x, y, 'ro')

        ax = fig.add_subplot(1, 2, 2)
        plt.plot(fp, tp, '.b-')
        ax.set_title('Area under ROC curve: ' + str(score))
        plt.axis((0, 1, 0, 1))
        plt.show()
        plt.pause(0.005)
        plt.close()
        
    return score

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

def discretize_gt(gt):
	import warnings
	warnings.warn('can improve the way GT is discretized')
	return gt/255

def compute_nss(sal_map, fix_binary):
    
    sal_map, fix_binary = adjust_image_size(sal_map, fix_binary)
    
    N_fixations = np.sum(fix_binary, axis = (0,1))
    
    sal_norm = (sal_map - np.mean(sal_map)) / np.std(sal_map)

    NSS = np.sum(np.multiply(sal_norm, fix_binary), axis = (0,1)) / N_fixations
    
    return NSS


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
   
    DATASET_NAME = 'mit1003'
    AUC_SCORE1 =[]
    AUC_SCORE2 =[]
    CC_S = []
    SIM_S = []
    MIT_KLDiv_S = []
    infogain_S = []
    NSS_S = []
   
    for image in stimu(DATASET_NAME):
    
        saliency_map = utils.saliency_map(DATASET_NAME, image)
        fixation_map = utils.fixation_map(DATASET_NAME, image)
        #show_map(DATASET_NAME, image,  showFixMap = True, showSalMap = True, showqlMap = True,
        #        wait_time=500, plotMaxDim = 1600)
        #show_scanpath(DATASET_NAME, image, subj, animation=True, plotMaxDim=1000, wait_time=500)
        #score1 = AUC_Judd(saliency_map[:682, :682], fixation_map[:682, :682], jitter=True, toPlot=False) 
        #print('Ground_Truth:', score1)
        #AUC_SCORE1.append(score1)
    
        saliency_map_ql = utils.saliency_map_ql(DATASET_NAME, image)
        score2 = AUC_Judd(saliency_map_ql, fixation_map, jitter=True, toPlot=False) 
        print('Saliency_MAP_QL:', score2)
        AUC_SCORE2.append(score2)
    
        saliency_map = utils.saliency_map(DATASET_NAME, image)
        h,w = saliency_map.shape
        saliency_map = resize(saliency_map, (682, 1024))
        saliency_map_ql = utils.saliency_map_ql(DATASET_NAME, image)
        saliency_map_ql = resize(saliency_map_ql, (682,1024))
        CC_Score = CC(saliency_map_ql, saliency_map)
        print('Similarity:', CC_Score)
        CC_S.append(CC_Score)
    
        saliency_map = utils.saliency_map(DATASET_NAME, image)
        h,w = saliency_map.shape
        saliency_map = resize(saliency_map, (682, 1024))
        saliency_map_ql = utils.saliency_map_ql(DATASET_NAME, image)
        saliency_map_ql = resize(saliency_map_ql, (682,1024))
        SIM_Score = SIM(saliency_map_ql, saliency_map)
        print('SIM:', SIM_Score)
        SIM_S.append(SIM_Score)
    
        saliency_map = utils.saliency_map(DATASET_NAME, image)
        saliency_map = resize(saliency_map, (682, 1024))
        fixation_map = utils.fixation_map(DATASET_NAME, image)
        fixation_map = resize(fixation_map, (682, 1024))
        saliency_map_ql = utils.saliency_map_ql(DATASET_NAME, image)
        saliency_map_ql = resize(saliency_map_ql, (682,1024))
        MIT_KLDiv_Score = kldiv(saliency_map_ql, saliency_map)
        print('KLDiv:', MIT_KLDiv_Score)
        MIT_KLDiv_S.append(MIT_KLDiv_Score)

        saliency_map = utils.saliency_map(DATASET_NAME, image)
        saliency_map = resize(saliency_map, (682, 1024))
        fixation_map = utils.fixation_map(DATASET_NAME, image)
        fixation_map = resize(fixation_map, (682, 1024))
        saliency_map_ql = utils.saliency_map_ql(DATASET_NAME, image)
        saliency_map_ql = resize(saliency_map_ql, (682,1024))
        infogain_Score = compute_information_gain(saliency_map_ql, saliency_map, baseline=[])
        print('Information Gain', infogain_Score)
        infogain_S.append(infogain_Score)

        saliency_map = utils.saliency_map(DATASET_NAME, image)
        saliency_map = resize(saliency_map, (682, 1024))
        fixation_map = utils.fixation_map(DATASET_NAME, image)
        fixation_map = resize(fixation_map, (682, 1024))
        saliency_map_ql = utils.saliency_map_ql(DATASET_NAME, image)
        saliency_map_ql = resize(saliency_map_ql, (682,1024))
        NSS_Score = compute_nss(saliency_map_ql, saliency_map)
        print('NSS', NSS_Score)
        NSS_S.append(NSS_Score)


    Mean_AUC_SCORE1 = np.mean(AUC_SCORE1)
    print('----------------')
    print('Mean_AUC_SCORE1:', Mean_AUC_SCORE1)
    print('----------------')
    
    Mean_AUC_SCORE2 = np.mean(AUC_SCORE2)
    print('----------------')
    print('Mean_AUC_SCORE2', Mean_AUC_SCORE2)
    print('----------------')
    
    Mean_CC_Scores = np.mean(CC_S)
    print('----------------')
    print('Mean_CC_Scores', Mean_CC_Scores)
    print('----------------')
    
    Mean_SIM_Scores = np.mean(SIM_S)
    print('----------------')
    print('Mean_SIM_Scores', Mean_SIM_Scores)
    print('----------------')
    
    Mean_MIT_KLDiv_Scores = np.mean(MIT_KLDiv_S)
    print('----------------')
    print('Mean_MIT_KLDiv_Scores:', Mean_MIT_KLDiv_Scores)
    print('----------------')

    Mean_infogain_Scores = np.mean(infogain_S)
    print('----------------')
    print('Mean_infogain_Scores:', Mean_infogain_Scores)
    print('----------------')
    
    Mean_NSS_Scores = np.mean(NSS_S)
    print('----------------')
    print('Mean_NSS_Scores:', Mean_NSS_Scores)
    print('----------------')
    

        #for image in stimu(DATASET_NAME):
        #    saliency_map = utils.saliency_map(DATASET_NAME, image)
        #    saliency_map_ql = utils.saliency_map_ql(DATASET_NAME, image)
        # 
        #     plot_information_gain(saliency_map, ax=None, color_range = None, image=None, frame=False,
        #                    thickness = 1.0, zoom_factor=1.0, threshold=0.05, rel_levels=None,
        #                    alpha=0.5, color_offset = 0.25, plot_color_bar=True)
        #    plt.show()
 
    '''
    DATASET_NAME = 'MIT1003'
    AUC_SCORE1 =[]
    AUC_SCORE2 =[]
    CC_S = []
    SIM_S = []
    MIT_KLDiv_S = []
    infogain_S = []
    NSS_S = []

    for image in stimu(DATASET_NAME):
    
        saliency_map = utils.saliency_map(DATASET_NAME, image)
        fixation_map = utils.fixation_map(DATASET_NAME, image)
        
        saliency_map_ITT = utils.saliency_map_ITT(DATASET_NAME, image)
        score2 = AUC_Judd(saliency_map_ITT, fixation_map, jitter=True, toPlot=False) 
        print('Saliency_MAP_ITT:', score2)
        AUC_SCORE2.append(score2)
    
        saliency_map = utils.saliency_map(DATASET_NAME, image)
        h,w = saliency_map.shape
        saliency_map = resize(saliency_map, (682, 1024))
        saliency_map_ITT = utils.saliency_map_ITT(DATASET_NAME, image)
        saliency_map_ITT = resize(saliency_map_ITT, (682,1024))
        CC_Score = CC(saliency_map_ITT, saliency_map)
        print('Similarity:', CC_Score)
        CC_S.append(CC_Score)
    
        saliency_map = utils.saliency_map(DATASET_NAME, image)
        h,w = saliency_map.shape
        saliency_map = resize(saliency_map, (682, 1024))
        saliency_map_ITT = utils.saliency_map_ITT(DATASET_NAME, image)
        saliency_map_ITT = resize(saliency_map_ITT, (682,1024))
        SIM_Score = SIM(saliency_map_ITT, saliency_map)
        print('SIM:', SIM_Score)
        SIM_S.append(SIM_Score)
    
        saliency_map = utils.saliency_map(DATASET_NAME, image)
        saliency_map = resize(saliency_map, (682, 1024))
        fixation_map = utils.fixation_map(DATASET_NAME, image)
        fixation_map = resize(fixation_map, (682, 1024))
        saliency_map_ITT = utils.saliency_map_ITT(DATASET_NAME, image)
        saliency_map_ITT = resize(saliency_map_ITT, (682,1024))
        MIT_KLDiv_Score = kldiv(saliency_map_ITT, saliency_map)
        print('KLDiv:', MIT_KLDiv_Score)
        MIT_KLDiv_S.append(MIT_KLDiv_Score)

        saliency_map = utils.saliency_map(DATASET_NAME, image)
        saliency_map = resize(saliency_map, (682, 1024))
        fixation_map = utils.fixation_map(DATASET_NAME, image)
        fixation_map = resize(fixation_map, (682, 1024))
        saliency_map_ITT = utils.saliency_map_ITT(DATASET_NAME, image)
        saliency_map_ITT = resize(saliency_map_ITT, (682,1024))
        infogain_Score = compute_information_gain(saliency_map_ITT, saliency_map, baseline=[])
        print('Information Gain', infogain_Score)
        infogain_S.append(infogain_Score)

        saliency_map = utils.saliency_map(DATASET_NAME, image)
        saliency_map = resize(saliency_map, (682, 1024))
        fixation_map = utils.fixation_map(DATASET_NAME, image)
        fixation_map = resize(fixation_map, (682, 1024))
        saliency_map_ITT = utils.saliency_map_ITT(DATASET_NAME, image)
        saliency_map_ITT = resize(saliency_map_ITT, (682,1024))
        NSS_Score = compute_nss(saliency_map_ITT, saliency_map)
        print('NSS', NSS_Score)
        NSS_S.append(NSS_Score)


    Mean_AUC_SCORE1 = np.mean(AUC_SCORE1)
    print('----------------')
    print('Mean_AUC_SCORE1:', Mean_AUC_SCORE1)
    print('----------------')
    
    Mean_AUC_SCORE2 = np.mean(AUC_SCORE2)
    print('----------------')
    print('Mean_AUC_SCORE2', Mean_AUC_SCORE2)
    print('----------------')
    
    Mean_CC_Scores = np.mean(CC_S)
    print('----------------')
    print('Mean_CC_Scores', Mean_CC_Scores)
    print('----------------')
    
    Mean_SIM_Scores = np.mean(SIM_S)
    print('----------------')
    print('Mean_SIM_Scores', Mean_SIM_Scores)
    print('----------------')
    
    Mean_MIT_KLDiv_Scores = np.mean(MIT_KLDiv_S)
    print('----------------')
    print('Mean_MIT_KLDiv_Scores:', Mean_MIT_KLDiv_Scores)
    print('----------------')

    Mean_infogain_Scores = np.mean(infogain_S)
    print('----------------')
    print('Mean_infogain_Scores:', Mean_infogain_Scores)
    print('----------------')
    
    Mean_NSS_Scores = np.mean(NSS_S)
    print('----------------')
    print('Mean_NSS_Scores:', Mean_NSS_Scores)
    print('----------------')
    '''
