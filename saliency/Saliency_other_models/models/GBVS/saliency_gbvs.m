
function [smap] = saliency_gbvs(input_image,image_path)


% get saliency map:
smap = gbvs_fast(input_image); 
smap = smap.master_map_resized;

end

