function [feature featuresize featurescale ii jj] = sample_from_L2(L2,feature_spatial_sizes,feature_scaledepth)
% Sample features from layer L2 

% INPUTS: 
% Layer L2

% OUTPUT:
% feature of scale depth "feature_scaledepth" centered at a random scale
% and at random spatial position. The size of the feature is taken also
% randomly selected in feature_spatial_sizes

[a b c d]=size(L2);
Numberofscales=c+1;
NumbofOrient=d;

featuresize=feature_spatial_sizes(randi([1 length(feature_spatial_sizes)]));

scalelist=(feature_scaledepth+1)/2:Numberofscales-(feature_scaledepth+1)/2;      %  select random central scale of the feature
s=scalelist(randi([1 length(scalelist)]));
  
ii=ceil( (a-featuresize+1)*rand(1));                                        %  select random spatial position of feature
jj=ceil( (b-featuresize+1)*rand(1));
feature=L2(ii:ii+featuresize-1,jj:jj+featuresize-1,s-(feature_scaledepth-1)/2:s+(feature_scaledepth-1)/2,:);
ma=max(max(feature,[],4),[],3);
feature=feature.*(feature==reshape(repmat(ma,1,NumbofOrient*feature_scaledepth),featuresize,featuresize,feature_scaledepth,NumbofOrient));
featurescale=s;

end