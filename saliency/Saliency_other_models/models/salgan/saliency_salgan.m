function [ smap ] = saliency_salgan( input_image, input_path )
	
	mkdir('images'); %input_image=imresize(input_image,[192 256]);
	imwrite(input_image,'images/default.jpg');
	system('./call.sh');
    
	smap=imread('saliency/default.jpg');
	smap=mat2gray(smap);
    delete('saliency/default.jpg');
    delete('images/default.jpg');
end

