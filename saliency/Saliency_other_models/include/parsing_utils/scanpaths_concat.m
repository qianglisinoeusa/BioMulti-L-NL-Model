function [ scanpath ] = scanpaths_concat( scanpaths)

    scanpath = [];
    for i=1:length(scanpaths)
       single_scanpath = scanpaths{i};
       scanpath = [scanpath; single_scanpath];
    end
    
end





