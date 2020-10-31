
function [smap] = saliency_aws(input_image,image_path)

resiz_param = 0.5;
sp=2.1;

% get saliency map:
smap = AWS(image_path,resiz_param,sp); 

end

