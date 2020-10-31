
function [smap] = saliency_pqft(input_image,image_path)

if size(input_image,3)<3
    input_image(:,:,2)=input_image(:,:,1);
    input_image(:,:,3)=input_image(:,:,1);
end

% get saliency map:
params{1}='gaussian'; %disk, gaussian
params{2}=size(input_image,2);
params{3}='color'; %color,grayscale

addpath(genpath('qtfm'));


[~,~,smap] = PQFT(input_image,input_image,params{1},params{2},params{3});

end
