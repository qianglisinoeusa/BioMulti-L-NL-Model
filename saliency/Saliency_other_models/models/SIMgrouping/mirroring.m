function mc = mirroring(w,n)
% Mirrors edges of the wavelet plane as a form of edge padding.
%
% outputs:
%   mc: padded wavelet plane
%
% inputs:
%   w: wavelet plane
%   n: extent of padding

[i j] = size(w);
A     = flipud(w(2:(2+n-1),:));    % top padding
B     = flipud(w(i-n:i-1,:));      % bottom padding
mc    = [A;w;B];                   % add padding

A  = fliplr(mc(:,2:(2+n-1)));      % left padding
B  = fliplr(mc(:,j-n:j-1));        % right padding
mc = [A mc B];                     % add padding

end
