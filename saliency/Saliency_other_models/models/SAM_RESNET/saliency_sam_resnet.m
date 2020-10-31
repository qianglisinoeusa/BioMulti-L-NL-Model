function [ smap ] = saliency_sam_resnet( input_image, input_path )
	
	mkdir('tmp_img');
	imwrite(input_image,'tmp_img/default.jpg');
	system('./call_resnet.sh');
	smap=imread('predictions/default.jpg');
    delete('predictions/default.jpg');
	smap=mat2gray(smap);
	rmdir('tmp_img','s');
end

