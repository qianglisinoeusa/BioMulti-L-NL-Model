function gauss=gauss2D(sigma,orient,aspect,norm,pxsize,x0,y0)
%function gauss=gauss2D(sigma,orient,aspect,norm,pxsize,x0,y0)
%
% This function produces a numerical approximation to Gaussian function with
% variable aspect ratio.
% Parameters:
% sigma  = standard deviation of Gaussian envelope, this in-turn controls the
%          size of the result (pixels)
% orient = orientation of the Gaussian clockwise from the vertical (degrees)
% aspect = aspect ratio of Gaussian envelope (0 = no "width" to envelope, 
%          1 = circular symmetric envelope)
% norm   = 1 to normalise the gaussian so that it sums to 1
%        = 0 for no normalisation (gaussian has max value of 1)
%          Optional, default value is 1.
% pxsize = the size of the filter.
%          Optional, if not specified size is 5*sigma.
% x0,y0  = location of the centre of the gaussian.
%          Optional, if not specified gaussian is centred in the middle of image.
if nargin<4
  norm=1;
end
if nargin<5
  pxsize=odd(5*sigma);
end
if nargin<6
  x0=0;
  y0-0;
end
[x y]=meshgrid(-fix(pxsize/2):fix(pxsize/2),fix(-pxsize/2):fix(pxsize/2));
 
% Rotation 
orient=-orient*pi/180;
x_theta=(x-x0)*cos(orient)+(y-y0)*sin(orient);
y_theta=-(x-x0)*sin(orient)+(y-y0)*cos(orient);

gauss=exp(-.5*( ((x_theta.^2)./(sigma.^2)) ...
				+ ((y_theta.^2)./((aspect*sigma).^2)) ));
if norm, 
  gauss=gauss./sum(sum(gauss)); 
end