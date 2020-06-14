function displaygabors(Gabor)
% This code displays images of the Gabor filters
ct=0;
for (i=1:size(Gabor,1))
    for (j=1:size(Gabor,2))
        ct=ct+1;
        subplot(size(Gabor,1),size(Gabor,2),ct)
        imagesc(Gabor{i,j});colormap(gray);        
    end
end
suptitle('Multi-Scale, Multi-Orien, Multi-RFsize, Multi-Wavelength')

end