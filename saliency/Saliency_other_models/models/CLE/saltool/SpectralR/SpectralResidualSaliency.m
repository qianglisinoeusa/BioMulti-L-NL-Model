function salMap = SpectralResidualSaliency(inImage)
% SpectralResidualSaliency - computes a salience map with the spectral
%                            residual method
%
% Synopsis
%   salMap = SpectralResidualSaliency(inImage)
%
% Description
%   The function computes a salience map with the spectral residual method.
%   The code is adapted from that published by X.Hou at 
%   http://www.klab.caltech.edu/~xhou/projects/spectralResidual/spectralresidual.html
%   The SR method provides comparable performance to other methods but at 
%   a lower computational complexity end it is easy to code
%
%
% Inputs ([]s are optional)
%   (matrix) inImage      the original image
%
% Outputs ([]s are optional)
%   (matrix) salMap          the saliaency map
%   ....
%
% Examples
%   sMap = SpectralResidualSaliency(originalImage);
%
%
% Requirements
%   Image Processing toolbox
%   Statistical toolbox

% References
%   Hou, X., Zhang, L.: Saliency detection: A spectral residual approach. 
%                       In: Proceedings CVPR ?07, vol. 1, pp. 1-8 (2007)
%   
%
%
% Changes
%   20/01/2011  First Edition
%
% Preparing the image 
inImg = im2double(rgb2gray(inImage));
[rows cols]=size(inImg);
inImg = imresize(inImg, [64, 64], 'bilinear');

% The actual Spectral Residual computation: just 5 Matlab lines!
myFFT = fft2(inImg); 
myLogAmplitude = log(abs(myFFT));
myLogAmplitude(isinf(myLogAmplitude))=min(min(myLogAmplitude(~isinf(myLogAmplitude)))); %log(0)=-Inf
myPhase = angle(myFFT);
mySpectralResidual = myLogAmplitude - imfilter(myLogAmplitude, fspecial('average', 3), 'replicate'); 
saliencyMap = abs(ifft2(exp(mySpectralResidual + 1i*myPhase))).^2;

% After Effect
saliencyMap = imfilter(saliencyMap, fspecial('disk', 3));

% Resizing from 64*64 to the original size
saliencyMap = mat2gray(saliencyMap);
saliencyMap = imresize(saliencyMap, [rows cols], 'bilinear');

salMap=im2double(saliencyMap);
end