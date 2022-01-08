import numpy as np

def div_norm_metric(im1,im2):

# VISIBILITY_OF_TARGET computes the perceptual difference between 2 images
# (target+background versus background) according to the Teo&Heeger metric 
# fitted as in the paper:
#
#    Divisive Normalization image Quality Metric Revisited
#
# The metric applies the non-linear divisive normalization (with the 
# optimal parameters) for the two color images and applies the optimal
# summation model to the difference vector.
#
# The distance, d, computed by the metric is linearly translated to 
# DMOS values (in the range [25,100]) using the fit of the LIVE and
# TID databases.
# 
# The program returns an intermediate result: the spatial distortion map 
# obtained before pooling accross space. This map gives an idea on how the
# different regions in the target contribute to the global visibility.
#
# USE: [d,DMOS,spatial_map]=visibility_of_target(target+background,background);

im_yuv1,pyr1,r1,ind = standard_V1_model_general(im1)
im_yuv2,pyr2,r2,ind = standard_V1_model_general(im2)

delta_r=np.abs(r1-r2)

d_fe,spatial_dist_map=summation_model_fe(np.abs(delta_r[1:-np.prod(ind[-1,:]),:]),ind,4.5,2.18)

d_fe=d_fe*np.numel(delta_r[1:-np.prod(ind[-1,:]),:])/(3*np.numel(im1[:,:,1]))

DMOS=4.864880793450431e+004*d_fe+25.84
return d_fe, DMOS, spatial_dist_map
