
function [smap] = saliency_simpsal(input_image,image_path)

if size(input_image,3) < 3
    input_image(:,:,2)=input_image(:,:,1);
    input_image(:,:,3)=input_image(:,:,1);
end

%set output parameters
%output_folder = 'output';
%output_extension = '.png';
%[in_folder,image_name,extension]=fileparts(image_path);
%image_name_noext = remove_extension(image_name);
%output_image = [image_name_noext output_extension];
%experiment_name = 'image_name_noext';
image_name=image_path;

p = default_fast_param;
p.blurRadius = 0.02;     % e.g. we can change blur radius 


% get saliency map:
%smap = simpsal(input_image,p);

smap = simpsal(image_name);

smap = 255*mat2gray(smap);



%write output image on output folder
%imwrite(uint8(smap),[output_folder '/' output_image]);
%save([output_folder '/' output_image '.mat'],'smap');

end

