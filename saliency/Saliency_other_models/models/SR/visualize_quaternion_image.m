function [eigenangle,eigenaxis,logmod,img_eigenangle,img_eigenaxis,img_logmod]=visualize_quaternion_image(I,do_figures)
  % VISUALIZE_QUATERNION_IMAGE provides implementations for visualization
  %   of 2-D Quaternion images. The implemented visualizations are 
  %   according to the (poster) visualizations in "HYPERCOMPLEX AUTO- AND 
  %   CROSS-CORRELATION OF COLOR IMAGES" by Stephen J. Sangwine and 
  %   Todd A. Ell.
  %
  % In the comments:
  %   S(.) is the scalar part (real part of the quaternion)
  %   V(.) is the vector part (imaginary part of the quaternion)
  %   abs  is the Modulus aka absolute value
  %
  % Output:
  %   eigenangle       The eigenangle aka phase
  %   eigenaxis        The eigenaxis
  %   logmod           The log-normalized modulus
  %   img_eigenangle   Image/visualization for the eigenangle
  %   img_eigenaxis    Image/visualization for the eigenaxis
  %   img_logmod       Image/visualization for the (log-normalized) modulus
  %
  % @author: B. Schauerte
  % @date:   2011
  % @url:    http://cvhci.anthropomatik.kit.edu/~bschauer/

  % Copyright 2011 B. Schauerte. All rights reserved.
  % 
  % Redistribution and use in source and binary forms, with or without 
  % modification, are permitted provided that the following conditions are 
  % met:
  % 
  %    1. Redistributions of source code must retain the above copyright 
  %       notice, this list of conditions and the following disclaimer.
  % 
  %    2. Redistributions in binary form must reproduce the above copyright 
  %       notice, this list of conditions and the following disclaimer in 
  %       the documentation and/or other materials provided with the 
  %       distribution.
  % 
  % THIS SOFTWARE IS PROVIDED BY B. SCHAUERTE ''AS IS'' AND ANY EXPRESS OR 
  % IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
  % WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
  % DISCLAIMED. IN NO EVENT SHALL B. SCHAUERTE OR CONTRIBUTORS BE LIABLE 
  % FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
  % CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
  % SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR 
  % BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
  % WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
  % OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
  % ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
  % 
  % The views and conclusions contained in the software and documentation
  % are those of the authors and should not be interpreted as representing 
  % official policies, either expressed or implied, of B. Schauerte.
  
  if nargin < 2, do_figures = true; end

  % Display the phase / eigenangle, i.e. phi = tan^{-1}(abs(V(I)) / S(I)):
  %   Please note that if it is a pure quaternion, then the phase angle
  %   is undefined (division by zero). 
  %   [In this case, the image will appear green]
  eigenangle=angle(I)/(2*pi);
  img_eigenangle=hsv2rgb(cat(3,eigenangle,ones(size(I)),ones(size(I))));
  if do_figures
    figure('name','eigenangle(I)');
    imshow(img_eigenangle);
  end
  
  % Display the Eigenaxis, i.e. mu = V(i) / abs(V(i)):
  vi=v(I);
  ai=abs(vi);
  nvi=vi ./ ai;
  unvi=unit(nvi);
  eigenaxis=cat(3,x(unvi),y(unvi),z(unvi));
  img_eigenaxis=eigenaxis;
  if do_figures
    figure('name','eigenaxis(I)');
    imshow(img_eigenaxis);
  end
  
  % Display the Modulus in log-grayscale, i.e. M = log(1 + abs(I)) / log(1 + max(abs(I))):
  %   (Modulus aka absolute value)
  maxabs=max(abs(I(:)));
  logmod=log(1 + abs(I)) / log(1 + maxabs);
  img_logmod=mat2gray(logmod);
  if do_figures
    figure('name','log-Magnitude');
    imshow(img_logmod);
  end
end

  
  
