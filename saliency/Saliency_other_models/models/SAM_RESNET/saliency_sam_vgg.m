function [ smap ] = saliency_sam_vgg( input_image, input_path )
	
	mkdir('tmp_img');
	imwrite(input_image,'tmp_img/default.jpg');
	system('./call_vgg.sh');
	smap=imread('predictions/default.jpg');
    delete('predictions/default.jpg');
	smap=mat2gray(smap);
	rmdir('tmp_img','s');
end

