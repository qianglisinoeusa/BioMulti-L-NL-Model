import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from my_rgb2yuv import my_rgb2yuv
from my_yuv2rgb import my_yuv2rgb
from skimage.util import montage
from color_space_trans import RGB2YUV, YUV2RGB
from scipy.io import loadmat
from alfa_beta_wav import alfa_beta_wav
import pyrtools as pyr
from buildWpyr import buildWpyr, make_grid_coeff_ql

# Rrtina stimuli
rgb_img = imread('I04.BMP')

nf, nc, capas=rgb_img.shape

n_rep_fil=np.ceil(nf/256)
n_rep_col=np.ceil(nc/256)

nf_amp=256*n_rep_fil
nc_amp=256*n_rep_col

#  (1) Linear stage
# 
#     1.1- Chromatic transform: RGB to Opponent Luminance-RG-YB
#          representations (im_yuv)

yuv_img = RGB2YUV(np.double(rgb_img))
rbg_img_b = YUV2RGB(yuv_img)

plt.figure()
mulchann_imgs = montage([np.double(rgb_img)/255, yuv_img/255, rbg_img_b/255], grid_shape=(1,3), multichannel=True)
plt.imshow(mulchann_imgs)
plt.axis('off')

plt.figure()
yuv_imgs = montage([rgb_img[:,:,1], yuv_img[:,:,0],yuv_img[:,:,1],yuv_img[:,:,2]], grid_shape=(1,4))
plt.imshow(yuv_imgs, cmap='gray')
plt.axis('off')
#plt.show()

##
#  (2) Non-linear stage
# 
#     2.1 Non-linear divisive normalization of each CSF filtered chromatic channel (r)
# 

params = loadmat('h_256_se_0_25_so_3_sxy_0_25_umbral_500_bits_6.mat');
#print(sorted(params.keys()))
#print(params['h_sparse'])

# Parameters
color=1;
Ao_y = 40;
Ao_uv = 35;
sigma_hv_y = 1.5;
sigma_hv_uv = 0.5;
fact_diag=0.8;
theta = 6;    #Exponent that controls the sharpness of the CSF
escalas=4;    #Total number of scales in the wavelet transform

# Excitation-inhibition exponent and regularization parameter  
g1=1.7;
beta=12;
A = np.array([ [Ao_y,Ao_uv], [fact_diag*Ao_y, fact_diag*Ao_uv] ])
s = np.array([sigma_hv_y,sigma_hv_uv])
#alfa(scale,orientation,chromatic_channel)
[alfas,betas] = alfa_beta_wav(A,s,theta,beta,escalas)

pyr_corto = np.zeros((54, 3))
#Non-linear responses in each 256*256 block
for i in range(1, np.int(n_rep_fil)):
    for j in range(1,np.int(n_rep_col)):
        bloque=yuv_img[(i-1)*255+1:(i-1)*255+256,(j-1)*255+1:(j-1)*255+256,:]
        
        for capa in range(0, 3):
           pyr_corto = buildWpyr(bloque[:,:,capa], scales=4, order=3)
           #pyr.imshow(pyr_corto[(3,1)])
           #plt.show()
           grid_rep = make_grid_coeff_ql(pyr_corto,num_scales=4,normalize=True)
           plt.imshow(np.flipud(grid_rep), cmap='gray')
           plt.draw()
           plt.show()
        '''
        r2 = non_linear_response(pyr_corto,ind_corto,alfas,g1,betas,h_sparse,color)
        
        for capa in range(1,4):
            for banda in range(1,14):

                B=pyrBand(r2(:,capa),ind_corto,banda)
                BANDS_im2(banda).capa(capa).ind(i,j).B = B

                B_w=pyrBand(pyr_corto(:,capa),ind_corto,banda)
                BANDS_im2_w(banda).capa(capa).ind(i,j).B = B_w
		'''

