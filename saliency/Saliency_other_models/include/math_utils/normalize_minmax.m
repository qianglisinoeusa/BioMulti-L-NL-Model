
function [map_out] = normalize_minmax(map_in,min_in, max_in)


if nargin < 2
max_in = max(map_in(:));
min_in = min(map_in(:));
end

map_out    = (map_in - min_in)/(max_in - min_in);




end
