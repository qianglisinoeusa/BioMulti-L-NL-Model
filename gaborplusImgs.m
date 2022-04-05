clear all; clc;
im =imread('zebra.jpeg');
im_1= rgb2gray(im);

im2 =imread('bird.jpg');
im_2= imresize(rgb2gray(im2),[size(im_1, 1) size(im_1, 2)]);

figure, imshow(im_1), axis off; set(gcf, 'color', 'w');
% 
% 
% clear all; clc;
% im =imread('zebra.jpeg');
% im_1= im;
% im2 =imread('bird.jpg');
% im_2= imresize(im2,[size(im_1, 1) size(im_1, 2)]);
% figure, imshow(im_1+im_2), axis off;
% 

numFrames = 12; 
sf = 2.0;
sigma = 13;
widthOfGrid = 300;  % size of the Gabor patch in pixels
ang = 45;
tiltInRadians = ang*pi/180; % The tilt of the grating in radians.
spatialFrequency = sf/16; % How many periods/cycles are there in a pixel?
radiansPerPixel = spatialFrequency*(2*pi); % = (periods per pixel) * (2 pi radians per period)
% Compute ramp
halfWidthOfGrid = widthOfGrid/2;
widthArray=(-halfWidthOfGrid):halfWidthOfGrid;  % widthArray is used in creating the meshgrid.
[x, y] = meshgrid(widthArray, widthArray);
a=sin(tiltInRadians);
b=cos(tiltInRadians);

ramp = (b*x + a*y);
% gaussian
gaussian=exp(-(((x-90)/sigma).^2)-(((y-90)/sigma).^2));

background=ones(301,301)*0.5;
for i=1:numFrames
    phase=(i/numFrames)*2*pi;
    % grating
    grating = sin(radiansPerPixel*ramp-phase);
    % gabor
    gabor = grating.*gaussian;
    figure(1), imshow(gabor+background), axis off; 
end
set(gcf, 'color', 'w');


for i=1:numFrames
    phase=(i/numFrames)*2*pi;
    % grating
    grating = sin(radiansPerPixel*ramp-phase);
    % gabor
    gabor = grating.*gaussian;
    %imGabor = uint8(255 * mat2gray((gabor+double(imresize(im_1,[size(gabor,1) size(gabor,1)])))));
    gabors=gabor;
    imG = gabors*255+imresize(double(im_1),[size(gabor,1) size(gabor,1)]);
    figure(1), subplot(221),imshow(gabor+background), axis off; subplot(223),imshow((imresize(im_1,[size(gabor,1) size(gabor,1)]))), axis off;
    subplot(224), imshow(uint8(imG)), axis off;
    pause(0.3)
end

