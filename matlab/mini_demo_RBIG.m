%Add Retina-LGN-V1-ongoing package into matlab path

%Load any image

load images_80.mat

r_xy=AdjancyCorrPixel(im1);
r_x2y=Adjancy2CorrPixel(im1);
r_x4y=Adjancy4CorrPixel(im1);
   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% De-corrlation with RBIG2018
% Addpath RBIG2008 into matlab
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
