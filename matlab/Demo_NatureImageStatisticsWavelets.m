% ======================================================================
%  Wavelets based on nature image statistcs Version 1.0
%  Copyright(c) 2020  Qiang Li
%  All Rights Reserved.
%  qiang.li@uv.es
% ----------------------------------------------------------------------
% Permission to use, copy, or modify this software and its documentation
% for educational and research purposes only and without fee is here
% granted, provided that this copyright notice and the original authors'
% names appear on all copies and supporting documentation. This program
% shall not be used, rewritten, or adapted as the basis of a commercial
% software or hardware product without first obtaining permission of the
% authors. The authors make no representations about the suitability of
% this software for any purpose. It is provided "as is" without express
% or implied warranty.


%%Add dependent toolbox path
addpath(genpath('/home/qiang/QiangLi/Matlab_Utils_Functional/matlab_human-vision-model_utils/Retina-LGN-V1-models-OnGoing/numerical-tours-master/matlab'));

%%Statistics of the Wavelets Coefficients of Natural Images

n = 256*2;
M = rescale( load_image('lena', n) );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Try use steerbale instead of perform_wavelet_transf function
Jmin = 4;
MW = perform_wavelet_transf(M,Jmin, +1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

MW1 = MW(1:n/2,n/2+1:n);

%low order statistics
v = max(abs(MW1(:)));
k = 20;
t = linspace(-v,v,2*k+1);
h = hist(MW1(:), t);

clf;
subplot(1,2,1);
imageplot(MW1);
subplot(1,2,2);
bar(t, h/sum(h)); axis('tight');
axis('square');

%%high order statistics
T = .03;
MW1q = floor(abs(MW1/T)).*sign(MW1);
nj = size(MW1,1);

t = 2; % 
C = reshape(MW1q([ones(1,t) 1:nj-t],:),size(MW1));

options.normalize = 1;
[H,x,xc] = compute_conditional_histogram(MW1q,C, options);

q = 8; % width for display
H = H((end+1)/2-q:(end+1)/2+q,(end+1)/2-q:(end+1)/2+q);
clf;
imagesc(-q:q,-q:q,max(log(H), -5)); axis image;
colormap gray(256);

options.normalize = 0;
[H,x,xc] = compute_conditional_histogram(MW1q,C, options);
H = H((end+1)/2-q:(end+1)/2+q,(end+1)/2-q:(end+1)/2+q);

clf;
contour(-q:q,-q:q,max(log(H), -5), 20, 'k'); axis image;
colormap gray(256);
