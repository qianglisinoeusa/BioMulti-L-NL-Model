function [ smap ] = saliency_bmsalt( input_image, input_path )
	if size(input_image,3)<3
		input_image(:,:,2)=input_image(:,:,1);
		input_image(:,:,3)=input_image(:,:,1);
	end
	imwrite(input_image,'default.png');
	system('/usr/bin/python3 saliency.py -i default.png');

	smap=imread('default_saliency.png');
	smap=mat2gray(smap);
	delete 'default_saliency.png';
end



