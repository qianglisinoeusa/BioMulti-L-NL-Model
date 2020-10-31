function [wFF]=filter_definitions_V1_simple(sigma,wFF)
%Define weights, based on Gabor functions, for V1 simple cells
angles=[0:22.5:179];
wavel=1.5*sigma;
aspect=1./sqrt(2);

if nargin<2
  k=0;
else
  k=size(wFF,1); %append new filters onto those already defined
end
for phase=0:90:270
  for angle=angles
	k=k+1;
	%define Gabor RF, and split into positive and negative parts
	gabor=gabor2D(sigma,angle,wavel,phase,aspect);
	gabor=single(gabor);
	norm=sum(sum(abs(gabor)));
	gabor=gabor./norm;
	wFF{k,1}=max(0,gabor); %ON channel (+ve part)
	wFF{k,2}=max(0,-gabor);%OFF channel (-ve part)
  end
end
