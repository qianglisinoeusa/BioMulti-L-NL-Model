% create 2-D sinusoidal gratings to be used in analysis of selectivity 
% and invariance. Output is in 1-D vector format.

function grating=creategratings(patchsize,freqvalue,orvalue,phasevalue)

%freqvalue:  frequency of grating where 1 means one cycle in patch
%orvalue:    orientation of grating where 0 means horizontal "stripes"
%phasevalue: phase of grating where 0 means sinusoidal

%create matrices that give the different x and y values
x=[0:patchsize-1]/patchsize;
y=x;
[xm,ym]=meshgrid(x,y);

%rotate x and y values according to the desired orientation
zm=sin(orvalue).*xm+cos(orvalue).*ym;

%compute sin and cos functions
grating2d=sin(zm*freqvalue*2*pi+phasevalue);

%normalize to unit norm
grating2d=grating2d/(norm(grating2d,'fro')+.00001);

%output in vector format
grating=reshape(grating2d,[patchsize^2 1]);

return;
