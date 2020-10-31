%
% call spectral residual aprroach with resolution fixes by author
% [1] X. Hou and L. Zhang, "Saliency Detection: A Spectral Residual
%  Approach", in CVPR, 2007.

function saliency_map=SR(imgname,fr)
    im=im2double(imread(imgname));
    %saliency_map=spectral_saliency_multichannel(im,fr,'fft:residual');
    saliency_map=spectral_saliency_multichannel(im,[64 64],'fft:residual');
end