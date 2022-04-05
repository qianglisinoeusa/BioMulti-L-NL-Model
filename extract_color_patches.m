clear all; clc;clc
%% patches all you need
img = imread('bird.jpg');
img = imresize(imread('bird.jpg'), [256 256]);

patches=im2colcube(img, [32 32], 1);
pat = reshape(patches, 32,32,3,64);

for i=1:(size(pat,4))
    figure(1), imshow(pat(:,:,:,i)),pause(0.2)
end


function color_patches=extract_color_patches(img, patch_size)

    img = imresize(imread(img));
    patches=im2colcube(img, [patch_size patch_size], 1);
    pat = reshape(patches, patch_size,patch_size,3,size(patches,2));
    for i=1:(size(pat,4))
        color_patches= pat(:,:,:,i);    
    end
end





