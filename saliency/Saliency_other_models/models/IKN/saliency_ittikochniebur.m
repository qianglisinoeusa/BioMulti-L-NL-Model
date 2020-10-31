
function [smap] = saliency_ittikochniebur(input_image,image_path)

gbvs_install

% get saliency map:
smap = ittikochmap(input_image); 

smap=smap.master_map;

end

