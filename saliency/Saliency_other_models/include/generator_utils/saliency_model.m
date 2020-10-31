
function [smap,scanpath,smaps] = saliency_model(model_name,input_image_path)

disp(input_image_path);

%read image
input_image = imread(input_image_path);

%set output size parameters
[M, N, C] = size(input_image);

% get saliency map:
switch nargout(model_name)
	case 1
		[smap] = feval(model_name, input_image, input_image_path);
		smap = edit_smap(smap,M,N);
	case 2
		[smap,scanpath] = feval(model_name, input_image, input_image_path);
		smap = edit_smap(smap,M,N);
	case 3
		[smap,scanpath,smaps] = feval(model_name, input_image, input_image_path);
		smap = edit_smap(smap,M,N);
		for g=1:size(smaps,3)
			esmaps(:,:,g)=edit_smap(smaps(:,:,g),M,N);
		end	
		smaps=esmaps;
	otherwise
		[smap] = feval(model_name, input_image, input_image_path);
		smap = edit_smap(smap,M,N);
end



end

function [smap] = edit_smap(smap, M, N)
	if nargin < 2, M=size(smap,1); N=size(smap,2); end

	%convert NaN to zeros
	smap(find(isnan(smap)))=0;
	    
	%from 0 to 1, then from 1 to 255
	smap = 255*mat2gray(smap); %255*normalize_minmax(smap);

	%resize to original input sizes
	if size(smap,1) ~= M || size(smap,2) ~= N
	    smap = imresize(smap,[M N]);
	end

end

