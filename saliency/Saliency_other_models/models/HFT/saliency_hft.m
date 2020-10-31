
function [smap] = saliency_hft(input_image,image_path)


if size(input_image,3)<3
    input_image(:,:,2)=input_image(:,:,1);
    input_image(:,:,3)=input_image(:,:,1);
end
input_image=double(input_image)./255;

addpath(genpath('qtfm'));

[smap] = HFT(input_image);

end
