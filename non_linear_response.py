'''

def resp_norm_stage_lin_nonlin(x,Hn,Hd,a,b,jacob):
    
    RESP_NORM_STAGE_lin_nonlin is the same as RESP_NORM_STAGE but it returns 
    not only the final nonlinear output, but also the intermediate linear stage. 
    RESP_NORM_STAGE computes the responses and Jacobian of a substractive/divisive neural stage
     
    The response vector "y" given the input vector "x" is
    
                     x - a*Hn*x
            y = ----------------------
                    b + Hd*abs(x)
    
    And each element of the Jacobian matrix is:
    
                            delta_{ij} - a*Hn_{ij}             (x_i - sum_k a*H_{ik}*x_k) * Hd_{ij}
      nabla_R_{ij} =  ----------------------------------  -  -------------------------------------------
                         b + sum_k Hd_{ik} abs(x_k)           ( b + sum_k Hd_{ik} abs(x_k) )^2
    
     [ y , y_interm, nablaL, nablaR, sx ] = resp_norm_stage(x,Hn,Hd,a,b,comp_jacob)
    
    
           x = input vector
    
           Hn and Hd = interaction kernels at the numerator and denominator
                       (see make_conv_kernel for Gaussian kernels).
                       The sum of the rows in these kernels is normalized to one        
    
           a = norm of the rows in the substractive interaction
    
           b = saturation constant in the divisive normalization
    
           if comp_jacobian == 1 -> computes the Jacobian (otherwise it only
           computes response).
    
    
    from __future__ import division
    from __future__ import absolute_import
    from __future__ import print_function
    import sys
    import os
    import numpy as np
    
    sx = np.sign(x)
    
    num = x - a*Hn*x
    div = b + Hd*np.abs(x)

    nablaL = np.eye(len(x)) - a*Hn

    # xx = x - a*Hn*x
    y = np..divide(num,div)

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

'''
from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import sys
import os
import numpy as np
import pyrtools as pyr

def non_linear_response(pyr,ind,alfas,exp,beta,h_sparse,color):

# NON_LINEAR_RESPONSE computes the divisive normalization for orthogonal wavelets.
#
#                |a.*w|.^g
#       r = --------------------
#             b + H * |a.*w|.^g
#
# Where the columns of the matrix 'w' contain the wavelet transform of each
# chromatic channel (using buildwpyr)
#
# 'a' and 'b' are generated with alfa_beta_wav, and have the following structure:
# Rows            (1) -> Scale             (from fine to coarse)
# Columns         (2) -> Orientation       (horizontal, vertical, diagonal)
# Third dimension (3) -> Chromatic channel (Y, U, V)
#
# And H is a sparse matrix generated with kernel_h_ort_hc
#
# USE:
#
# r   = non_linear_response(pyr,ind,a,g,b,H,color)


# Datos
[escalas orientaciones canales] = alfas.shape
capas = pyr.shape[1]
if color==0:
   pyr=pyr[:,1]
   capas=1s

# Vector de respuestas
r = pyr

for capa in range(1, capas+1): # Y, U, V
        
    # Creamos un vector 'a' similar a 'r' sin la componente
    # de continua pero de una capa
    a = np.zeros((pyr.shape[0]-np.prod(ind[end,:]),1))
    b = np.ones ((pyr.shape[0]-np.prod(ind[end,:]),1))          
    
    for escala in range (1, escalas+1): # De mas alta frecuencia a mas baja
        for orientacion in range(1,orientaciones+1): # H V D

            # Calculamos la subbanda a procesar segun la escala y la
            # orientacion donde nos encontramos
            banda = orientaciones * (escala - 1) + orientacion

            # Calculamos los indices para dicha subbanda
            indices = pyr.pyr_coeffs([ind,banda])

            # Aplicamos la CSF y la no linealidad a cada subbanda                    
            a[indices] = np.power((np.abs ((pyr[indices,capa] .* alfas[escala,orientacion,capa]))), exp)
            b[indices] = np.power(( np.dot(b[indices], beta[escala,orientacion,capa]), exp))
        
    # En cada capa calculamos la Normalizacion Divisiva
    r[1:a.shape(0),capa] = np.divide(a,( b + h_sparse * a ))
    r[:,capa] = np.dot(np.sign(pyr.pyr_coeffs(:,capa)) .* np.abs(r[:,capa]))

return r