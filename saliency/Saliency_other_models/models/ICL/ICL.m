function mySMap=ICL(nomeimg,fr)

%fr non ten efecto. O autor reduce as imaxes a 64 na maxima dimension

load AW.mat;

%% Reading image
inIm = im2double(imread(nomeimg));
inIm = imresize(inIm, 64/size(inIm, 2));
sze=size(inIm);

if numel(sze)==2 
    inImg(:,:,1)=inIm(:,:);
    inImg(:,:,2)=inIm(:,:);
    inImg(:,:,3)=inIm(:,:); 
else
    inImg=inIm;
    
end


[imgH, imgW, imgDim] = size(inImg);

%% Building Saliency Saliency
myEnergy = im2Energy(inImg, W);
mySMap = vector2Im(myEnergy, imgH, imgW);

%% Visualization
mySMap = mySMap.^2;
mySMap = imfilter(mySMap, fspecial('gaussian', [8, 8], 8));


% figure(1);
% subplot(1,2,1)
% imshow(mySMap,[]);
% subplot(1,2,2)
% imshow(inImg);
% 
% %% (Optional) Showing basis
% figure(2);
% showBasis(A);

end