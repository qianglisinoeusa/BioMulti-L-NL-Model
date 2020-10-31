function [median_gazes] = get_mediangazes(path_to_gbg_folder)
%% Number of gazes in which most stimulus have data -> usually about 5,10,20...
    %as the distribution of available data per gaze tends to be logarithmic (not normal) we'll use the median
 dirs=sort_nat(listpath_dir(path_to_gbg_folder));
 for i=1:length(dirs)
 	num(i)=length(listpath([path_to_gbg_folder '/' dirs{i}]));
 end
 if length(num)<1
     median_gazes=0; 
     return
 end
 if mod(length(num),2)==0
     num=[num,num(end)];
 end
 found=num(find(num==median(num)));
 median_gazes=length(found);

end

