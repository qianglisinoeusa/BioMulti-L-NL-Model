
function [smap] = saliency_achanta(input_image,image_path)

if size(input_image,3)<3
    input_image(:,:,2)=input_image(:,:,1);
    input_image(:,:,3)=input_image(:,:,1);
end


% get saliency map:
smap = Achanta(input_image); 

smap=normalize_minmax(smap);

end
