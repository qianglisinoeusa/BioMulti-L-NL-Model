function [ smap ] = saliency_ml_net( input_image, input_path )
	
	%cd '/media/dberga/DATA/repos/BIOvsDL/modelos/ML_NET';
	mkdir('tmp_img');
	imwrite(input_image,'tmp_img/default.jpg');
	system('./call.sh');
	smap=imread('default.jpg');
	delete('default.jpg');
	smap=mat2gray(smap);
	rmdir('tmp_img','s');
end

