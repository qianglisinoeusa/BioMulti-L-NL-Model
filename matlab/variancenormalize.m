% Normalizes variances of patches
% Data given as a matrix where each patch is one column vectors

function Xnorm=variancenormalize(X);

patchnorms=sum(X.^2);
%compute 10% lower quantile
%and define epsilon as this quantile
[patchnormssorted,dummy]=sort(patchnorms,'ascend');
epsilon=patchnormssorted(round(size(X,2)/10));
Xnorm=X./(ones(size(X,1),1)*patchnorms+epsilon).^(1/2); 