function saliencyMap=SRScor(nome,fr)
%% Read image from file e redimensionamos



%O autor redimensiona a [64 64]
inImg = imresize(imread(nome), [64, 64], 'bilinear');
%inImg = imresize(imread(nome),fr, 'bilinear');
si=size(inImg);
S=zeros(si(1),si(2));

%% Spectral Residual

for indb=1:3
    myFFT = fft2(inImg(:,:,indb));
    myLogAmplitude = log(abs(myFFT));
    myPhase = angle(myFFT);
    mySpectralResidual = myLogAmplitude - imfilter(myLogAmplitude, fspecial('average', 3), 'replicate');
    saliencyMap = abs(ifft2(exp(mySpectralResidual + 1i*myPhase))).^2;
    
    S=S+saliencyMap;
end

%% After Effect
S = imfilter(S, fspecial('disk', 3));
saliencyMap = mat2gray(saliencyMap);
saliencyMap = filter2(fspecial('gaussian',63,64*0.04),saliencyMap);

end