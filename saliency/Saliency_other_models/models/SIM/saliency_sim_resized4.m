
function [smap] = saliency_sim(input_image,image_path)

if size(input_image,3) < 3
    input_image(:,:,2)=input_image(:,:,1);
    input_image(:,:,3)=input_image(:,:,1);
end

[m n p]      = size(input_image);
window_sizes = [13 26];                          % window sizes for computing center-surround contrast
wlev         = min([7,floor(log2(min([m n])))]); % number of wavelet planes
gamma        = 2.4;                              % gamma value for gamma correction
srgb_flag    = 1;                                % 0 if img is rgb; 1 if img is srgb
rsz=2;


% get saliency map:
smap = SIM_resized(input_image, window_sizes, wlev, gamma, srgb_flag,rsz); 

end

