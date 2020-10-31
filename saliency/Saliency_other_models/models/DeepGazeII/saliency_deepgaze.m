function [ smap ] = saliency_deepgaze( input_image, input_path )
	if size(input_image,3)<3
		input_image(:,:,2)=input_image(:,:,1);
		input_image(:,:,3)=input_image(:,:,1);
	end
	imwrite(input_image,'default.png');
	system('/usr/bin/python3 demo.py');

	smap=imread('saliency_default.png');
	smap=mat2gray(smap);
	delete 'saliency_default.png';
end

