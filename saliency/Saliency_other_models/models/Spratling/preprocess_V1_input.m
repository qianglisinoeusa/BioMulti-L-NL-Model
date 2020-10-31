function [X]=preprocess_V1_input(I,sigma)
response_gain=2*pi;
[a,b,z]=size(I);

LoG=-fspecial('log',odd(9*sigma),sigma);
%To avoid using Image Processing toolbox try:
%LoG=gauss2D(sigma-0.0001,0,1,1,odd(9*sigma))-gauss2D(sigma+0.0001,0,1,1,odd(9*sigma));

%normalise weights
tmp=LoG;
tmp(find(tmp<0))=0;
LoG=LoG./sum(sum(tmp));

for t=1:z %at each time step
  %calculare LGN neuron responses to input image
  Xonoff=conv2(I(:,:,t),LoG,'same');

  %apply gain to response
  Xonoff=(response_gain.*Xonoff);

  %apply saturation to response
  Xonoff=tanh(Xonoff);

  %split into ON and OFF channels
  Xon=Xonoff;
  Xon(find(Xon<0))=0;
  Xoff=-Xonoff;
  Xoff(find(Xoff<0))=0;
  
  X{1}(:,:,t)=Xon;
  X{2}(:,:,t)=Xoff;
end
