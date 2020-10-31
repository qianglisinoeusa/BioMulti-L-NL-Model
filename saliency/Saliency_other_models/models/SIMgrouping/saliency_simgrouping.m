
function [smap] = saliency_simgrouping(input_image,image_path)

if size(input_image,3) < 3
    input_image(:,:,2)=input_image(:,:,1);
    input_image(:,:,3)=input_image(:,:,1);
end

[m n p]      = size(input_image);
window_sizes = [17 37];                 % window sizes for computing normalized center contrast
gamma        = 2.4;                     % gamma value for gamma correction
srgb_flag    = 1;                       % 0 if img is rgb; 1 if img is srgb

% factor by which to resize image: 
% image should NOT be resized before calling SIM, 
% as the size of the image is related to the CSF.
rsz = 2;



% get saliency map:
smap = SIM(input_image, window_sizes, gamma, srgb_flag, rsz); 

end

