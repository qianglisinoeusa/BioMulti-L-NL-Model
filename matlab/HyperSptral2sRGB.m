function hyperSptral2sRGB()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The code explained how to convert hypersptral image to sRGB.
% The hypersptral dataset from Forster., University of Machester.
% Download and Tutorials:https://personalpages.manchester.ac.uk/staff/d.h.foster/Tutorial_HSI2RGB/Tutorial_HSI2RGB.html
% The code adapted from Forsted et al,.

% Author:Q.LI
% Unit:University of Valencia
% Data: 2020

% Reference:
% Foster, D.H., & Amano, K. (2019). Hyperspectral imaging in color vision research: tutorial. 
% Journal of the Optical Society of America A, 36, 606-627.
% Run Environment: /home/qiang/QiangLi/Matlab_Utils_Functional/matlab_human-vision-model_utils/Retina-LGN-V1-models-OnGoing
% Data Info:The Dataset store in the Images/Foster2002_HyperImgDatabase.

% ref4_scene4.mat is a test hyperspectral image consisting of an array of spectral reflectances of size 255 x 355 x 33. The first two 
% coordinates represent spatial dimensions (pixels), in row-column format, and the third coordinate represents wavelength (400, 410, ..., 720 nm),
% as in Fig. 1. For ease of handling in this tutorial, the image has been reduced spatially to about a quarter of the size of those downloadable here.
% illum_25000.mat, illum_6500.mat, and illum_4000.mat are three illuminant spectra, each vectors of length 33 (i.e. 400, 410, ..., 720 nm), representing 
% the spectra of blue skylight with correlated colour temperature (CCT) 25000 K, daylight with CCT 6500 K, and evening sunlight with CCT 4000 K [6]. 
% Formulae for generating daylight spectra are provided in [6].
% xyzbar.mat contains the CIE 1931 colour-matching functions [6]. Tablulated values are also available here.
% XYZ2sRGB_exgamma.m is a routine for converting tristimulus values XYZ to the default RGB colour space sRGB, but without gamma correction .
% A copy of the document IEC_61966-2-1.pdf explaining the full conversion of XYZ to sRGB  is also included.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; close; clc;
% 3. Reflectance data
tic
% The size 255*335*33(400, 410, , 720 nm in step 10nm);
load 'Images/Foster2002_HyperImgDatabase/ref4_scene4.mat';

size(reflectances)

% Check middle slices (560nm)
slice = reflectances(:,:,17);
figure; imagesc(slice); colormap('gray'); axis square tight, axis off; brighten(0.5);
hold on,
drawrectangle(39,100,20,[],'r-'); axis equal tight;
title(['wavelength:', num2str(560), 'nm']);

z = max(slice(100, 39));
slice_clip = min(slice, z)/z;
figure; imagesc(slice_clip.^0.4); colormap('gray'); axis square tight, axis off;
title(['rescale-','wavelength:', num2str(560), 'nm']);


% Spectral reflectance row-column coordinates (141, 75).

reflectance = squeeze(reflectances(141, 75,:));
figure; plot(400:10:720, reflectance);
xlabel('wavelength, nm');
ylabel('unnormalized reflectance');


reflectances = reflectances/max(reflectances(:)); 


% 4. Radiance data
load 'Images/Foster2002_HyperImgDatabase/illum_25000.mat';
radiances_25000 = zeros(size(reflectances)); 

for i = 1:33,
  radiances_25000(:,:,i) = reflectances(:,:,i)*illum_25000(i);
end

radiance_25000 = squeeze(radiances_25000(141, 75, :));
figure; plot(400:10:720, radiance_25000, 'b-');
xlabel('wavelength, nm');
ylabel('radiance, arbitrary units');
hold on;

load 'Images/Foster2002_HyperImgDatabase/illum_4000.mat';
radiances_4000 = zeros(size(reflectances)); % initialize array
for i = 1:33,
  radiances_4000(:,:,i) = reflectances(:,:,i)*illum_4000(i);
end
radiance_4000 = squeeze(radiances_4000(141, 75, :));
plot(400:10:720, radiance_4000, 'r-'); 

% 5. Converting hyperspectral data to CIE XYZ and sRGB representations
% make an RGB representation of the scene in ref4_scene4.mat  under a global
% illuminant with CCT 6500 K. Load the illuminant spectrum illum_6500 and obtain reflected radiances radiances_6500.

load 'Images/Foster2002_HyperImgDatabase/illum_6500.mat';
radiances_6500 = zeros(size(reflectances)); % initialize array
for i = 1:33,
  radiances_6500(:,:,i) = reflectances(:,:,i)*illum_6500(i);
end

% Convert the radiance data radiances into tristimulus values XYZ
radiances = radiances_6500;

[r c w] = size(radiances);
radiances = reshape(radiances, r*c, w);

% Load the CIE 1931 colour matching functions xyzbar and take the matrix product of 
% xyzbar with radiances to get the tristimulus values XYZ at each pixel. For simplicity XYZ is not normalized. 

load 'Images/Foster2002_HyperImgDatabase/xyzbar.mat';
XYZ = (xyzbar'*radiances')';


% Correct the shape of XYZ so that it represents the original 2-dimensional image with three planes (rather than a matrix of 3 columns),
% and ensure that the values of XYZ range within 0-1 (conventional normalization so that Y = 100 is unnecessary here). 

XYZ = reshape(XYZ, r, c, 3);
XYZ = max(XYZ, 0);
XYZ = XYZ/max(XYZ(:));


% 5.2. RGB colour image
% Call XYZ2RGB_exgamma function
cd ('Images/Foster2002_HyperImgDatabase');
RGB = XYZ2sRGB_exgamma(XYZ);
RGB = max(RGB, 0);
RGB = min(RGB, 1);

figure; imshow(RGB.^0.4, 'Border','tight');
axis square tight, axis off;
hold on,
drawrectangle(17,244,20,[],'r-'); axis equal tight;
title('sRGB Imaging');

% rescale
z = max(RGB(244,17,:));
RGB_clip = min(RGB, z)/z;
figure; imshow(RGB_clip.^0.4, 'Border','tight');
title('rescale sRGB Imaging');

toc
tile 

end