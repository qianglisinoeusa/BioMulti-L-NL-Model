%subroutine for creating gabors
%this routine is not really used much:
%it is *not* used in analysis of RF's but only in creating 
%a single gabor illustration of in Chapter 1.

function gaborf=gabor(windowsize,xo,yo,sigma,freq,type);

%size of window is [-5,5] x [-5,5]
%y-axis is now horizontal
%"type" controls whether is it is odd- or even-symmetric
%"windowsize" is size of window in pixels

g2d=zeros(windowsize);
m=(windowsize+1)/2;
fact=5/(windowsize-m);

for x=1:windowsize
for y=1:windowsize

if type=='o';
g2d(y,x)=exp(-(((x-m)*fact-xo)/sigma)^2-(((y-m)*fact-yo)/sigma)^2)*sin(((y-m)*fact-yo)*freq);
else
g2d(y,x)=exp(-(((x-m)*fact-xo)/sigma)^2-(((y-m)*fact-yo)/sigma)^2)*cos(((y-m)*fact-yo)*freq);
end

end
end

gaborf=reshape( g2d,[windowsize^2 1]);
