input_path = '/home/qiang/QiangLi/Python_Utils_Functional/FixaTons/WECSF/mit1003/smaps/';
save_path = '/home/qiang/QiangLi/Python_Utils_Functional/FixaTons/WECSF/mit1003/enhanced';
        
[ smap ] = wecsf(copy_to_path1, save_path)

function [ smap ] = wecsf(input_path, save_path )
	
	%system('call_wecsf.sh');
	srcFiles = dir('input_path');  

	for i = 1:length(srcFiles)
	    i
	    filename = strcat(input_path, srcFiles(i).name);
	    [filpath, names, ext] = fileparts(filename);
	    smap = imread(filename);
		smap=mat2gray(smap);
		wecsf_map = imfilter(smap, fspecial('gaussian', [8, 8], 8));
		name = strcat(save_path, names, ext) ;
    	imwrite(wecsf_map,name)  
end
