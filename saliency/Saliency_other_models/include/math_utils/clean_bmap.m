function [ bmap ] = clean_bmap(bmap, percentage )
    if ~exist('percentage','var') percentage = 0.9; end;
    
    bmap(bmap < percentage) = 0;
    bmap(bmap >= percentage) = 1;
end

