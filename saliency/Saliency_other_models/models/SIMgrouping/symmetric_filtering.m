function b = symmetric_filtering(a, h)
% Filter a with h, using symmetric filtering at the edges, where extreme
% edges are not mirrored.
%
% outputs:
%   b: filtered matrix
%
% inputs:
%   a: input matrix
%   h: filter

% Extent of padding is half radius of filter:
pad_sz = floor(length(h)/2);

a_padded = mirroring(a, pad_sz);                  % pad edges
b_padded = imfilter(a_padded, h, 'replicate');    % filter padded matrix
b        = b_padded((pad_sz + 1):(end - pad_sz),(pad_sz + 1):(end - pad_sz)); % remove padding

end
