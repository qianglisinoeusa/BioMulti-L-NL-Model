function [imsizefac,crop,max_radius,sigma]=common_param_values

imsizefac=10/1.6; %featureSize/1.6
sigma=4; %ceil([imsizefac/1.6]);
crop=ceil(5*max(sigma));
max_radius=ceil(5*max(sigma))+1;

