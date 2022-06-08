%%
%%
%%
%%
clear all; clc; close all;
img = rgb2gray(imread('demo1.jpg'));
imf=fftshift(fft2(double(img)));
impf=abs(imf).^2;

N=21;
f=-N/2:N/2-1;
imagesc(f,f,log10(impf)), axis xy, colormap gray

raPsd2d(img, 256)