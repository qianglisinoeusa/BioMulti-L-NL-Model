%
% call spectral residual aprroach with resolution fixes by author
% [1] X. Hou and L. Zhang, "Saliency Detection: A Spectral Residual
%  Approach", in CVPR, 2007.

function saliency_map=QDCT(imgname,fr)
    im=im2double(imread(imgname));
	if size(im,3)<3
	    im(:,:,2)=im(:,:,1);
	    im(:,:,3)=im(:,:,1);
	end

    %saliency_map=spectral_saliency_multichannel(im,fr,'fft:residual');
%    saliency_map=spectral_saliency_multichannel(im,[64 64],'fft:residual');
	saliency_map=spectral_saliency_multichannel(im,[64 64],'quat:dct');
end



  %   There are several methods (multichannel_method) to calculate the 
  %   multichannel saliency:
  %   'fft':          by default the same as 'fft:whitening'
  %   'fft:whitening' Uses spectral whitening to calculate the saliency of
  %                   each channel separately and then averages the result.
  %   'fft:residual'  Uses the spectral residual to calculate saliency of 
  %                   each channel separately and then averages the result.
  %   'dct'           Uses DCT-based image signatures to calculate saliency
  %                   of each channel separately and then averages the 
  %                   result.
  %   'quat:fft':     Converts the image into a quaternion-based 
  %                   representation, uses quaternion FFT/IFFT operations.
  %   'quat:dct'      Converts the image into a quaternion-based 
  %                   representation, uses quaternion DCT/IDCT operations.
  %   'quat:dct:fast' Same as 'quad:dct', but with a fixed image 
  %                   resolution of 64x48 and uses optimized .mex files for
  %                   faster calculation.
  %   [...]           some others, e.g., Itti-Koch and GBVS for reference
  % 
