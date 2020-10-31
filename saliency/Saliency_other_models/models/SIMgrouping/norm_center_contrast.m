function zctr = norm_center_contrast(X,orientation,window_sizes)
% returns normalized center contrast (NCC) for each coefficient of a wavelet plane
%
% outputs:
%   zctr: matrix of NCC values for each coefficient
% 
% inputs:
%   X: wavelet plane
%   window sizes: window sizes for computing NCC
%   orientation: wavelet plane orientation

center_size   = window_sizes(1);
surround_size = window_sizes(2);

% center masks for horizontal, vertical and diagonal orientations:
hhc = ones(1,center_size);
hvc = hhc';
hdc = ceil((diag(hhc) + fliplr(diag(hhc)))/4);

% surround masks for horizontal, vertical and diagonal orientations:
hhs = [ones(1,surround_size) zeros(1,center_size) ones(1,surround_size)];
hvs = hhs';
hds = diag(hhs) + fliplr(diag(hhs));

% horizontal orientation:
if orientation == 1   
    mean_cen  = imfilter(X,hhc,'symmetric')/(length(find(hhc~=0)));
    mean_sur  = imfilter(X,hhs,'symmetric')/(length(find(hhs~=0)));
    sigma_cen = imfilter((X - mean_cen).^2,hhc.^2,'symmetric')/(length(find(hdc~=0)));
    sigma_sur = imfilter((X - mean_sur).^2,hhs.^2,'symmetric')/(length(find(hds~=0)));        
% vertical orientation:
elseif orientation == 2
    mean_cen  = imfilter(X,hvc,'symmetric')/(length(find(hvc~=0)));
    mean_sur  = imfilter(X,hvs,'symmetric')/(length(find(hvs~=0)));
    sigma_cen = imfilter((X - mean_cen).^2,hvc.^2,'symmetric')/(length(find(hdc~=0)));
    sigma_sur = imfilter((X - mean_sur).^2,hvs.^2,'symmetric')/(length(find(hds~=0)));    
% diagonal orientation:
elseif orientation == 3
    mean_cen  = imfilter(X,hdc,'symmetric')/(length(find(hvc~=0)));
    mean_sur  = imfilter(X,hds,'symmetric')/(length(find(hvs~=0)));
    sigma_cen = imfilter((X - mean_cen).^2,hdc.^2,'symmetric')/(length(find(hdc~=0)));
    sigma_sur = imfilter((X - mean_sur).^2,hds.^2,'symmetric')/(length(find(hds~=0)));    
end

% compute normalized center contrast:
zctr = (sigma_cen.^2)./((sigma_cen).^2+(sigma_sur).^2 + 1e-12);

end