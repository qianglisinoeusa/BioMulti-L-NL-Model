function [ scanpath ] = pad_erase_scanpath( scanpath, limits )
    
        scanpath(:,1) = scanpath(:,1)-limits(3);
        scanpath(:,2) = scanpath(:,2)-limits(1);

end

