function [pim,zoff] = PolarTransform(img,zcenter,rs,thetas)
% function [pim,zoff] = PolarTransform(img,zcenter,rs,thetas)
% extract the polar transform of a section of img
% polar transform origin at zcenter(x+iy)
% rs(pixels),thetas(radians) are sets of radii and angles to sample
% if they are scalars, then the radii are 1:r
% and thetas is taken to be dtheta in
% thetas = linspace(0, 2*pi , round(2*pi/dtheta) ); thetas =
thetas(1:end-1)
% zoff are the coordinates of the sample points in the original image
(x+iy)
% REA 4/15-6/1/05
if isscalar(rs)
rs = 1:round(rs);
end

if isscalar(thetas)
thetas = linspace(0, 2*pi , round(2*pi/thetas) );
thetas = thetas(1: (end-1) );
end

[thetas_img,rs_img] = meshgrid(thetas,rs);
[xs,ys] = pol2cart(thetas_img,rs_img);

% offset the x,y coords to the appropriate location
zoff = complex(xs,ys) + zcenter;

% and extract the image data
pim = interp2(img,real(zoff),imag(zoff),'nearest');
