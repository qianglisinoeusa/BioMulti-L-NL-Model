function [  ] = alldata2baseline(alldata,folder_name,dataset_name,resolutions,pxva)


concat_scanpath_all=cell(1,length(alldata));
concat_scanpath_first_all=cell(1,length(alldata));

concat_scanpath_pergaze=cell(1,length(alldata));
concat_scanpaths_pergaze_sep=cell(1,length(alldata));

resolution=mode(resolutions);

for i=1:length(alldata)
    scanpaths=alldata{i};
    
    %erase first fixation and save its scanpath
    scanpaths_first=cell(1,length(scanpaths));
    for pp=1:length(scanpaths)
        if size(scanpaths{pp},1)>0
            scanpaths_first{pp}=scanpaths{pp}(1,:);
            scanpaths{pp}(1,:)=[];
        else
            scanpaths_first{pp}=[];
        end
    end
    
    %scanpaths_all
    scanpath_all = scanpaths_concat(scanpaths);
    scanpath_first_all = scanpaths_concat(scanpaths_first);
        
    scanpaths_pergaze = scanpaths_gazes(scanpaths);
    scanpaths_pergaze_sep = scanpaths_gazes_sep(scanpaths);
        
    
    if resolutions(i,1) ~= resolution(1) || resolutions(i,2) ~= resolution(2)
        
        %unequal coordinates? -> not valid for baseline, just discard
        
%         [scanpath_all(:,1),scanpath_all(:,2)]=resize_coords(scanpath_all(:,1),scanpath_all(:,2),resolutions(i,:),resolution);
%         [scanpath_first_all(:,1),scanpath_first_all(:,2)]=resize_coords(scanpath_first_all(:,1),scanpath_first_all(:,2),resolutions(i,:),resolution);
% 
%         scanpath_all(:,1)=round(scanpath_all(:,1));
%         scanpath_all(:,2)=round(scanpath_all(:,2));
%         scanpath_first_all(:,1)=round(scanpath_first_all(:,1));
%         scanpath_first_all(:,2)=round(scanpath_first_all(:,2));
%         
%         scanpath_all = limit_scanpath(scanpath_all,resolution);
%         scanpath_first_all = limit_scanpath(scanpath_first_all,resolution);

        
    else
        concat_scanpath_all{i}=scanpath_all;
        concat_scanpath_first_all{i}=scanpath_first_all;
        
    end
    
end


scanpath_baseline=scanpaths_concat(concat_scanpath_all);
scanpath_first_baseline=scanpaths_concat(concat_scanpath_first_all);


%bmaps_all
bmap_baseline = scanpath2bmap(scanpath_baseline,resolution);
fbmap_baseline = scanpath2bmap(scanpath_first_baseline,resolution);

%dmaps_all
dmap_baseline = normalize_minmax(zhong2012(bmap_baseline,pxva));
fdmap_baseline = normalize_minmax(zhong2012(fbmap_baseline,pxva));

if ~exist([folder_name '/baseline/' dataset_name],'file') mkdir([folder_name '/baseline/' dataset_name]); end %folder
if ~exist([folder_name '/fbaseline/' dataset_name],'file') mkdir([folder_name '/fbaseline/' dataset_name]); end %folder
imwrite(dmap_baseline,[folder_name '/baseline/' dataset_name '/' dataset_name '.png']);
imwrite(fdmap_baseline,[folder_name '/fbaseline/' dataset_name '/' dataset_name '.png']);
        
        
end

