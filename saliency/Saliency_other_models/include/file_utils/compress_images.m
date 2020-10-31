function compress_images(folder_images, folder_images_new)
    if nargin< 2, folder_images_new=[folder_images '_mod']; end
    
    cell_images=listpath(folder_images);
    mkdir(folder_images_new);
    for i=1:length(cell_images)
       input_image_path=[folder_images '/' cell_images{i}];
       output_image_path=[folder_images_new '/' remove_extension(cell_images{i}) '.jpg'];
       image=imread(input_image_path); %read
       image=imresize(image,[512 640]); %modify
       imwrite(image,output_image_path,'Quality',80);
    end
end