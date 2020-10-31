function saliencyMap=SRS(nome,fr)
%% Read image from file
inImg = im2double(rgb2gray(imread(nome)));

inImg = imresize(inImg, fr, 'bilinear');
%inImg = imresize(inImg, [64, 64], 'bilinear');

%% Spectral Residual
myFFT = fft2(inImg);
myLogAmplitude = log(abs(myFFT));
myPhase = angle(myFFT);
mySpectralResidual = myLogAmplitude - imfilter(myLogAmplitude, fspecial('average', 3), 'replicate');
saliencyMap = abs(ifft2(exp(mySpectralResidual + 1i*myPhase))).^2;

%% After Effect
saliencyMap = imfilter(saliencyMap, fspecial('disk', 3));
% saliencyMap = mat2gray(saliencyMap);
% imshow(saliencyMap);


end