clear all; clc; close all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Gaussian copula transform
% QiangLI
% University of Valencia
% Nov,8, 2021
% @copyriht
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Ns = 1000;
aux_X = randn(1,Ns);

X(1,:) = cos(aux_X);
X(2,:) = sinc(aux_X);

X = X + 0.2*randn(size(X));
X = [0.5 0.5;-0.5 0.5]*X;

figure, scatterhist(X(1,:),X(2,:),'Kernel','on','Color','k','Marker','.')
figure, scatterhist(ctransform(X(1,:)'),ctransform(X(2,:)'),'Kernel','on','Color','k','Marker','.')
figure, scatterhist(norminv(ctransform(X(1,:)')),norminv(ctransform(X(2,:)')),'Kernel','on','Color','k','Marker','.')

function [ x ] = ctransform(x)
% CTRANSFORM Copula transformation (empirical CDF)
%   cx = ctransform(x) returns the empirical CDF value along the first
%   axis of x. Data is ranked and scaled within [0 1] (open interval).
[~,x] = sort(x, 1);
[~,x] = sort(x, 1);
x = x / (size(x, 1) + 1);


