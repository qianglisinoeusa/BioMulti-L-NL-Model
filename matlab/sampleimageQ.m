function X = sampleimagesQ(samples, winsize);

% gathers patches from the grey-scale images, no preprocessing done yet
%
% INPUT variables:
% samples            total number of patches to take
% winsize            patch width in pixels
%
% OUTPUT variables:
% X                  the image patches as column vectors

%IMPORTANT: IMAGE DATA IS SUPPOSED TO BE IN DIRECTORY ./DATA/
%CHANGE THIS ON LINE 34 IF YOU HAVE MOVED THE DATA ELSEWHERE
  
%----------------------------------------------------------------------
% Gather rectangular image patches
%----------------------------------------------------------------------

% We have a total of 13 images.
dataNum = 20;

% This is how many patches to take per image
getsample = floor(samples/dataNum);

% Initialize the matrix to hold the patches
X = zeros(winsize^2,samples);

sampleNum = 1;  
for i=(1:dataNum)

  % Even things out (take enough from last image)
  if i==dataNum, getsample = samples-sampleNum+1; end
  
  % Load the image. Change the path here if needed.
  I = imread(['Images/Miscellaneous-USC-DataBase/misc/color_image' num2str(i) '.tiff']);

  % Transform to double precision
  I = double(I);

  % Normalize to zero mean and unit variance (optional)
  I = I-mean(mean(I));
  I = I/sqrt(mean(mean(I.^2)));
  
  % Sample patches in random locations
  sizex = size(I,2); 
  sizey = size(I,1);
  posx = floor(rand(1,getsample)*(sizex-winsize-2))+1;
  posy = floor(rand(1,getsample)*(sizey-winsize-1))+1;
  for j=1:getsample
    X(:,sampleNum) = reshape( I(posy(1,j):posy(1,j)+winsize-1, ...
			posx(1,j):posx(1,j)+winsize-1),[winsize^2 1]);
    sampleNum=sampleNum+1;
  end 
  
end



