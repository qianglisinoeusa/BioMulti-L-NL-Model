function lms = dkl2lms_d65(dkl)
% Convert from DKL color space to LMS color space assuming adaptation to
% D65 white.

% lms_gray - the LMS coordinates of the white point
lms_gray = [0.739876529525622   0.320136241543338   0.020793708751515];

mc1 = lms_gray(1)/lms_gray(2);
mc2 = (lms_gray(1)+lms_gray(2))/lms_gray(3);

M_lms_dkl = [ 1  1 0;
               1 -mc1 0;
              -1 -1 mc2 ];
          
M_dkl_lms = inv(M_lms_dkl);          
                    
lms = cm_colorspace_transform( dkl, M_dkl_lms );          
          
end
