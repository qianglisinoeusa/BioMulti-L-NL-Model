filename     = '3.jpg';
img          = imread(filename);
[m n p]      = size(img);
window_sizes = [17 37];                 % window sizes for computing normalized center contrast
gamma        = 2.4;                     % gamma value for gamma correction
srgb_flag    = 1;                       % 0 if img is rgb; 1 if img is srgb

% factor by which to resize image: 
% image should NOT be resized before calling SIM, 
% as the size of the image is related to the CSF.
rsz = 2;

% get saliency map:
smap = SIM(img, window_sizes, gamma, srgb_flag, rsz);

figure(1);
subplot(1,2,1); imshow(img);
subplot(1,2,2); imshow(smap,[]);
