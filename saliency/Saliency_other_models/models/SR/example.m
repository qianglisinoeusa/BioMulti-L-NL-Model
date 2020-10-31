%
% a typical example call for the main function
%
saliency_map=spectral_saliency_multichannel(im2double(imread('example-images/golden_retriever.jpg')),[48 64],'quat:dct:fast');