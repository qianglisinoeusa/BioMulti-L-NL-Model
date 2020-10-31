function smap = SIM(img, window_sizes, gamma, srgb_flag, rsz)
% returns saliency map for image
%
% outputs:
%   smap: saliency map for image
%
% inputs:
%   imag: input image
%   window sizes: window sizes for computing normalized center contrast; suggested 
%   value of [17 37]
%   wlev: # of wavelet levels
%   gamma: gamma value for gamma correction
%   srgb_flag: 0 if img is rgb; 1 if img is srgb

% replicate channel if image is greyscale:
[m,n,p] = size(img);
if p == 1
    img = repmat(img,[1,1,3]);
end

% convert opponent colour space of colour images:
img     = double(img);
opp_img = rgb2opponent(img, gamma, srgb_flag);

% resize:
nu_0    = rsz;
m_resiz = round(m/(2.^rsz));
n_resiz = round(n/(2.^rsz));
opp_img = imresize(opp_img,[m_resiz n_resiz],'bilinear');

% set # of wavelet levels:
mwlev = log2(2^ceil(log2(max([m_resiz n_resiz]))));
cwlev = 1:mwlev;
pdims = (2^mwlev)./(2.^(0:mwlev-1));
if any(pdims == 4)
    wlev = cwlev(pdims == 4);
else
    wlev = mwlev;
end

% generate swam for each channel:
rec(:,:,1) = SWAM_per_channel(opp_img(:,:,1),wlev,nu_0,'colour',window_sizes);
rec(:,:,2) = SWAM_per_channel(opp_img(:,:,2),wlev,nu_0,'colour',window_sizes);
rec(:,:,3) = SWAM_per_channel(opp_img(:,:,3),wlev,nu_0,'intensity',window_sizes);

% combine channels:
s_map = sqrt(sum(rec.^2,3));

% normalise:
s_map   = imresize(s_map,[m,n]);
map_max = max(s_map(:));
map_min = min(s_map(:));
smap    = floor(255*(s_map - map_min)/(map_max - map_min));

end

function rec = SWAM_per_channel(channel,wlev,nu_0,mode,window_sizes)
% returns saliency map for channel
%
% outputs:
%   rec: saliency map for channel
%
% inputs:
%   channel: opponent colour channel for which saliency map will be computed
%   wlev: # of wavelet levels
%   nu_0: offset of peak spatial frequency for CSF in case of image
%   resizing
%   mode: type of channel i.e. colour or intensity
%   window sizes: window sizes for computing normalized center contrast; suggested 
%   value of [17 37]

% perform grouplet transform on channels and apply ECSF:
[w wc] = GT(channel,wlev, nu_0, mode, window_sizes);

% reconstruct the image using inverse wavelet transform:
rec = IDWT(w,wc,size(channel,2),size(channel,1));
    
% normalize:
if sum(rec(:)) > 0
    rec = rec./sum(rec(:));
end

end

