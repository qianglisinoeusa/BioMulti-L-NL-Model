function displayFFTgabors(Gabor)
% This code displays images of the Gabor filters with FFT 
ct=0;
for (i=1:size(Gabor,1))
    for (j=1:size(Gabor,2))
        ct=ct+1;
        subplot(size(Gabor,1),size(Gabor,2),ct)
        imagesc(20*log(abs(fftshift(fft2(Gabor{i,j}))))); colormap(hot);
    end
end

end