
function [smap] = saliency_casd(input_image,image_path)

if size(input_image,3) < 3
    input_image(:,:,2)=input_image(:,:,1);
    input_image(:,:,3)=input_image(:,:,1);
end

% get saliency map:
smap_cell = saliency({image_path}); 

smap_struct=smap_cell{1};
smap = smap_struct.SaliencyMap;

end

