function [ scanpath ] = bmap2scanpath( bmap )

    [y,x]=find(bmap>0);
    scanpath=[y,x];

end

