function [ scanpath_out ] = resize_scanpath( scanpath_in , size_in, size_out)

    bmap_in=scanpath2bmap( scanpath_in,size_in);
    bmap_out=imresize(bmap_in,size_out,'nearest');
    scanpath_out=bmap2scanpath(bmap_out);
end

