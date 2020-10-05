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

    # xx = x - a*Hn*x;
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



