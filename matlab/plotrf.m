function plotrf( A, cols ,filename )

% display receptive field(s) or basis vector(s) for image patches
%
% A         the basis, with patches as column vectors
% cols      number of columns (x-dimension of grid)
% filename  where to save the image, If [], don't save but plot
%           In some versions of matlab, must use '' instead of []

global figurepath

%set colormap
colormap(gray(256));

%normalize each patch
A=A./(ones(size(A,1),1)*max(abs(A)));

% This is the side of the window
dim = sqrt(size(A,1));

% Helpful quantities
dimp = dim+1;
rows = floor(size(A,2)/cols);  %take floor just in case cols is badly specified

% Initialization of the image
I = ones(dim*rows+rows-1,dim*cols+cols-1);

%Transfer features to this image matrix
for i=0:rows-1
  for j=0:cols-1
    % This sets the patch
    I(i*dimp+1:i*dimp+dim,j*dimp+1:j*dimp+dim) = ...
         reshape(A(:,i*cols+j+1),[dim dim]);
  end
end

%Save of plot results
imagesc(I); 
axis equal
axis off
print('-dps',[figurepath,filename,'.eps'])




