function L1 = L1_layer(Gabor, A)
% L1 LAYER  (normalized dot product of gabor on local patches of image A at every possible locations and scales)

% INPUTS: 
% Gabor: cell array containing the Gabor filters
% A : image

% OUTPUT:
% Layer L1 with dimensions:   size(A,1) x size(A,2) x Numberofscales x NumbofOrient

Numberofscales=size(Gabor,1);
NumbofOrient=size(Gabor,2);
[m n]=size(A);


%   L1 LAYER  (normalized dot product of gabor on local patch at every possible location and scale)
L1=zeros(m,n,Numberofscales,NumbofOrient);
for (s=1:Numberofscales)                                                  % loop on scales    
for (o=1:NumbofOrient)                                                    % loop on orientations
dotprod=conv2(A,Gabor{s,o},'same');                                       % dot product of gabor with local patch at every position 


filtersize=size(Gabor{s,1},1);

normpatch=sqrt(conv2(ones(filtersize,1),ones(filtersize,1),A.^2,'same')); % norm of each patch

tmp= abs(dotprod./normpatch);   % normalize

tmp(1:(filtersize+1)/2-1,:)=zeros((filtersize+1)/2-1,n);  % cut border effect
tmp(:,1:(filtersize+1)/2-1)=zeros(m,(filtersize+1)/2-1);  % cut border effect
tmp(m-(filtersize-1)/2+1:m,:)=zeros((filtersize-1)/2,n);  % cut border effect
tmp(:,n-(filtersize-1)/2+1:n)=zeros(m,(filtersize-1)/2);  % cut border effect
L1(:,:,s,o)= tmp;
end
end

% set to zero values obtained on uniform (or almost uniform) image regions
L1(L1<10^-7)=0;
L1(isnan(L1))=0;
L1(isinf(L1))=0;

end