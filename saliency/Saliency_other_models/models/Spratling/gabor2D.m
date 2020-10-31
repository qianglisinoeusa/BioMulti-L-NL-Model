function gb=gabor2D(sigma,orient,wavel,phase,aspect,pxsize)
%function gb=gabor2D(sigma,orient,wavel,phase,aspect)
%
% This function produces a numerical approximation to 2D Gabor function.
% Parameters:
% sigma  = standard deviation of Gaussian envelope, this in-turn controls the
%          size of the result (pixels)
% orient = orientation of the Gabor clockwise from the vertical (degrees)
% wavel  = the wavelength of the sin wave (pixels)
% phase  = the phase of the sin wave (degrees)
% aspect = aspect ratio of Gaussian envelope (0 = no "width" to envelope, 
%          1 = circular symmetric envelope)
% pxsize = the size of the filter (optional). If not specified, size is 5*sigma.
if nargin<6
  pxsize=fix(5*sigma);
end
if mod(pxsize,2)==0, pxsize=pxsize+1; end %give mask an odd dimension
[x y]=meshgrid(-fix(pxsize/2):fix(pxsize/2),fix(-pxsize/2):fix(pxsize/2));
 
% Rotation 
orient=-orient*pi/180;
x_theta=x*cos(orient)+y*sin(orient);
y_theta=-x*sin(orient)+y*cos(orient);

phase=phase*pi/180;
freq=2*pi./wavel;

gb=exp(-0.5.*( ((x_theta.^2)/(sigma^2)) ...
			 + ((y_theta.^2)/((aspect*sigma)^2)) )) ...
   .* (cos(freq*y_theta+phase) - cos(phase).*exp(-0.25.*((sigma*freq).^2)));

if abs(sum(sum(gb)))./sum(sum(abs(gb)))>0.01
  disp('WARNING unbalanced Gabor');
  abs(sum(sum(gb)))./sum(sum(abs(gb)))
end