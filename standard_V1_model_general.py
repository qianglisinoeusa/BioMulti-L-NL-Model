from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import sys
import os
import numpy as np
import pyrtools as pyr
from scipy.io import loadmat

def standard_V1_model_general(im):

    # STANDARD_V1_MODEL_GENERAL computes the response at V1 for a color image
    # 
    # The parameters of the neural model come from indirect psychophysical
    # mesurements: the model is fitted to reproduce subjective distortion
    # opinion.
    # 
    # The iamge is assumed to be observed at a distance such as the sampling
    # frequency is 64 cycles/deg in each dimension.
    # 
    # Model structure:
    # ----------------
    # 
    # V1 neurons have (spatial) wavelet-like impulse response (receptive fields) with 
    # (spectral) broad band excitatory sensitivity (achromatic neurons) and opponent 
    # excitatory-inhibitory spectral sensitivities (Yellow-Blue, Red-Green).
    # Overall response of V1 neurons is adaptive (non-linear) and can be understood 
    # as a two stage process:
    # 
    #  (1) Linear stage
    # 
    #      1.1- Chromatic transform: RGB to Opponent Luminance-RG-YB
    #           representations (im_yuv)
    # 
    #      1.2- Linear wavelet transform of each chromatic channel (w)
    # 
    #  (2) Non-linear stage
    # 
    #      - Non-linear divisive normalization of each chromatic channel (r)
    # 
    # The output is a M * N rows * 3columns matrix with the three normalized
    # representations of each opponent color channel.
    # 
    # NOTE: The low frequency residual is not processed.
    # 
    # Syntax:
    # -------
    #
    # [im_yuv,w,r,ind]=standard_V1_model_general(im)
    #
    #     Input:  MM rows * NN cols * 3 channels RGB matrix
    #     Output: M*N rows * 3 columns matrix with the responses 
    #
    # If the number of rows or columns is not 256, the image is zero
    # padded up to the nearest 256 multiple since Divisive Normalization is
    # applied in every 256*256 block of the input image.

    #
    # Zero-padding to make the image size 256-multiple
    #

    nf, nc, capas=im.shape

    n_rep_fil=np.ceil(nf/256)
    n_rep_col=np.ceil(nc/256)

    nf_amp=256*n_rep_fil
    nc_amp=256*n_rep_col

    imm2=np.zeros((nf_amp,nc_amp,3))

    if capas==1
        imm2[1:nf,1:nc,1]=im
        imm2[1:nf,1:nc,2]=im
        imm2[1:nf,1:nc,3]=im
    else
        imm2[1:nf,1:nc,:]=im

    ##################################################################
    #     # Image in the linear opponent Luminance-YellowBlue-RedGreen
    #  1  # linear wavelet representation
    #     # LINEAR PERCEPTUAL RESPONSE (with no CSF)
    ##################################################################

    # Linear chromatic transform

    im_yuv=my_rgb2yuv(imm2)

    # 4-scale QMF Wavelet transform of each opponent color channel

    escalas=4 

        # The wavelet transform is computed in each 256*256 block of the image
        # (see below)  

    ##############################################################
    #     # Image in the linear opponent Luminance-YellowBlue-RedGreen
    #  2  # and non-linearly normalized wavelet representation
    #     # NON-LINEAR PERCEPTUAL RESPONSE (including CSF)
    ##############################################################  

    # Optimal parameters of divisive normalization (fitted to LIVE database -max rho-)
   
    # Divisive normalization kernel
    #sigma_e = 0.25
    #sigma_o = 3
    #sigma_xy = 0.25
    #umbral = 500
   
    params = loadmat('h_256_se_0_25_so_3_sxy_0_25_umbral_500_bits_6.mat')
    
    # CSF filters
    color=1
 
    Ao_y = 40
    Ao_uv = 35
    sigma_hv_y = 1.5
    sigma_hv_uv = 0.5
    fact_diag=0.8
    theta = 6
    
    #Excitation-inhibition exponent and regularization parameter  
    g1=1.7
    beta=12
   
    alfas,betas = alfa_beta_wav([Ao_y Ao_uvfact_diag*[Ao_y Ao_uv]],[sigma_hv_y sigma_hv_uv],theta,beta,escalas)
       
    #Non-linear responses in each 256*256 block
              
    for i in range(1, n_rep_fil):
        for j in range(1,n_rep_col):

            bloque=im_yuv((i-1)*255+1:(i-1)*255+256,(j-1)*255+1:(j-1)*255+256,:)
            
            for capa in range(1, 4):
               [pyr_corto(:,capa) ind_corto] = buildWpyr(bloque(:,:,capa),escalas)

            r2 = non_linear_response(pyr_corto,ind_corto,alfas,g1,betas,h_sparse,color)
            
            for capa in range(1,4):
                for banda in range(1,14):

                    B=pyrBand(r2(:,capa),ind_corto,banda)
                    BANDS_im2(banda).capa(capa).ind(i,j).B = B

                    B_w=pyrBand(pyr_corto(:,capa),ind_corto,banda)
                    BANDS_im2_w(banda).capa(capa).ind(i,j).B = B_w
   
    # Gathering the blocks together into a single wavelet-like vector

    [pyr ind] = buildWpyr(im_yuv(:,:,1),4)
    pyr_w=np.zeros([len(pyr) 3])
    pyr_r=np.zeros([len(pyr) 3])

    for capa in range(1,4):
        for banda in range(1,14):

            [IN] = pyrBandIndices(ind,banda)
            [BB] = pyrBand(pyr,ind,banda)

            BB2=[]
            aux2=[]
            BB2w=[]
            aux2w=[]

            for i in range(1, n_rep_fil):
                for j in range(1, n_rep_col):

                    BB = BANDS_im2(banda).capa(capa).ind(i,j).B
                    aux2 = [aux2 BB]
                    
                    BBw = BANDS_im2_w(banda).capa(capa).ind(i,j).B
                    aux2w = [aux2w BBw]
    
                BB2 = [BB2aux2]
                BB2w = [BB2waux2w]            
                aux2=[]
                aux2w=[]            
        
            r2_aux(IN)=BB2[:]
            w2_aux(IN)=BB2w[:]
            
        pyr_r[:,capa]=r2_aux
        pyr_w[:,capa]=w2_aux    

    w=pyr_w
    r=pyr_r       
    
    return im_yuv,w,r,ind