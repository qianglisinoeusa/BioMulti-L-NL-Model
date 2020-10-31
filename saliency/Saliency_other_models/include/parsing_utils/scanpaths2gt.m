function [] = scanpaths2gt(scanpaths,i_str,folder_name,dataset_name,resolution,pxva)

        %erase first fixation and save its scanpath
	scanpaths_comp=scanpaths;
        scanpaths_first=cell(1,length(scanpaths));
        for pp=1:length(scanpaths)
            if size(scanpaths{pp},1)>0
                scanpaths_first{pp}=scanpaths{pp}(1,:);
                scanpaths{pp}(1,:)=[];
            else
                scanpaths_first{pp}=[];
            end
        end

        scanpaths_pergaze = scanpaths_gazes(scanpaths);
        scanpaths_pergaze_sep = scanpaths_gazes_sep(scanpaths);
        scanpaths_comp_pergaze = scanpaths_gazes(scanpaths_comp);
        scanpaths_comp_pergaze_sep = scanpaths_gazes_sep(scanpaths_comp);
    
        %write
        if ~exist([folder_name '/scanpaths/' dataset_name],'file') mkdir([folder_name '/scanpaths/' dataset_name]); end %folder
        scanpath = scanpaths; save([folder_name '/scanpaths/' dataset_name '/' i_str '.mat'],'scanpath'); %scanpath

        for pp=1:length(scanpaths)
            %scanpath for participant pp
            scanpath_pp = scanpaths{pp};
	    scanpath_comp_pp = scanpaths_comp{pp};
            scanpath_first_pp = scanpaths_first{pp};

            %bmap for participant pp
            bmap_pp = scanpath2bmap(scanpath_pp,resolution);
            fbmap_pp = scanpath2bmap(scanpath_first_pp,resolution);

            %dmap for participant pp
            dmap_pp = normalize_minmax(zhong2012(bmap_pp,pxva));
            fdmap_pp = normalize_minmax(zhong2012(fbmap_pp,pxva));

            %distmap for participant pp
            distmap_pp = normalize_minmax(bmap2distmap(bmap_pp));

            %write
            if ~exist([folder_name '/scanpaths/' dataset_name '/pp/' num2str(pp)],'file') mkdir([folder_name '/scanpaths/' dataset_name '/pp/' num2str(pp)]); end %folder
            if ~exist([folder_name '/bmaps/' dataset_name '/pp/' num2str(pp)],'file') mkdir([folder_name '/bmaps/' dataset_name '/pp/' num2str(pp)]); end %folder
            if ~exist([folder_name '/dmaps/' dataset_name '/pp/' num2str(pp)],'file') mkdir([folder_name '/dmaps/' dataset_name '/pp/' num2str(pp)]); end %folder
	    if ~exist([folder_name '/distmaps/' dataset_name '/pp/' num2str(pp)],'file') mkdir([folder_name '/distmaps/' dataset_name '/pp/' num2str(pp)]); end %folder
            if ~exist([folder_name '/fdmaps/' dataset_name '/pp/' num2str(pp)],'file') mkdir([folder_name '/fdmaps/' dataset_name '/pp/' num2str(pp)]); end %folder
            scanpath = scanpath_comp_pp; save([folder_name '/scanpaths/' dataset_name '/pp/' num2str(pp) '/' i_str '.mat'],'scanpath'); %scanpath
            imwrite(im2uint8(bmap_pp),[folder_name '/bmaps/' dataset_name '/pp/' num2str(pp) '/' i_str '.png']);
            imwrite(im2uint8(dmap_pp),[folder_name '/dmaps/' dataset_name '/pp/' num2str(pp) '/' i_str '.png']);
            imwrite(im2uint8(distmap_pp),[folder_name '/distmaps/' dataset_name '/pp/' num2str(pp) '/' i_str '.png']);
            imwrite(im2uint8(fdmap_pp),[folder_name '/fdmaps/' dataset_name '/pp/' num2str(pp) '/' i_str '.png']);
        end

        %scanpaths_all
        scanpath_all = scanpaths_concat(scanpaths);
        scanpaths_first_all = scanpaths_concat(scanpaths_first);
        
        %bmaps_all
        bmap_all = scanpath2bmap(scanpath_all,resolution);
        fbmap_all = scanpath2bmap(scanpaths_first_all,resolution);

        %dmaps_all
        dmap_all = normalize_minmax(zhong2012(bmap_all,pxva));
        fdmap_all = normalize_minmax(zhong2012(fbmap_all,pxva));

        %distmaps_all
	distmap_all = normalize_minmax(bmap2distmap(bmap_all));

        %write
        if ~exist([folder_name '/bmaps/' dataset_name],'file') mkdir([folder_name '/bmaps/' dataset_name]); end %folder
        if ~exist([folder_name '/dmaps/' dataset_name],'file') mkdir([folder_name '/dmaps/' dataset_name]); end %folder
	if ~exist([folder_name '/distmaps/' dataset_name],'file') mkdir([folder_name '/distmaps/' dataset_name]); end %folder
        if ~exist([folder_name '/fdmaps/' dataset_name],'file') mkdir([folder_name '/fdmaps/' dataset_name]); end %folder
        imwrite(bmap_all,[folder_name '/bmaps/' dataset_name '/' i_str '.png']);
        imwrite(dmap_all,[folder_name '/dmaps/' dataset_name '/' i_str '.png']);
	imwrite(distmap_all,[folder_name '/distmaps/' dataset_name '/' i_str '.png']);
        imwrite(fdmap_all,[folder_name '/fdmaps/' dataset_name '/' i_str '.png']);

        for g=1:length(scanpaths_pergaze)
            g_str = num2str(g);

            %scanpaths_all_gbg
            scanpath_all_gbg = scanpaths_pergaze{g};
	    scanpath_comp_all_gbg = scanpaths_comp_pergaze{g};

            %bmaps_all_gbg 
            bmap_all_gbg = scanpath2bmap(scanpath_all_gbg,resolution);

            %dmaps_all_gbg 
            dmap_all_gbg = normalize_minmax(zhong2012(bmap_all_gbg,pxva));

	    %distmaps_all_gbg 
	    distmap_all_gbg = normalize_minmax(bmap2distmap(bmap_all_gbg));
            
            %write
            if ~exist([folder_name '/scanpaths/' dataset_name '/gbg' '/' g_str],'file') mkdir([folder_name '/scanpaths/' dataset_name '/gbg' '/' g_str]); end %folder
            if ~exist([folder_name '/bmaps/' dataset_name '/gbg' '/' g_str],'file') mkdir([folder_name '/bmaps/' dataset_name '/gbg' '/' g_str]); end %folder
            if ~exist([folder_name '/dmaps/' dataset_name '/gbg' '/' g_str],'file') mkdir([folder_name '/dmaps/' dataset_name '/gbg' '/' g_str]); end %folder
	    if ~exist([folder_name '/distmaps/' dataset_name '/gbg' '/' g_str],'file') mkdir([folder_name '/distmaps/' dataset_name '/gbg' '/' g_str]); end %folder
            scanpath = scanpath_comp_all_gbg; save([folder_name '/scanpaths/' dataset_name '/gbg/' g_str '/'  i_str '.mat'],'scanpath'); %scanpath
            imwrite(bmap_all_gbg,[folder_name '/bmaps/' dataset_name '/gbg/' g_str '/'  i_str '.png']);
            imwrite(dmap_all_gbg,[folder_name '/dmaps/' dataset_name '/gbg/' g_str '/'  i_str  '.png']);
            imwrite(distmap_all_gbg,[folder_name '/distmaps/' dataset_name '/gbg/' g_str '/'  i_str  '.png']);
        end
        
        for g=1:length(scanpaths_pergaze_sep)
            g_str = num2str(g);

            %scanpaths_all_gbg
            scanpath_all_gbg_sep = scanpaths_pergaze_sep{g};
	    scanpath_comp_all_gbg_sep = scanpaths_comp_pergaze_sep{g};
            
            %bmaps_all_gbg 
            bmap_all_gbg_sep = scanpath2bmap(scanpath_all_gbg_sep,resolution);
            
            %dmaps_all_gbg 
            dmap_all_gbg_sep = normalize_minmax(zhong2012(bmap_all_gbg_sep,pxva));

	    %distmaps_all_gbg 
            distmap_all_gbg_sep = normalize_minmax(bmap2distmap(bmap_all_gbg_sep));
            
            %write
            if ~exist([folder_name '/scanpaths/' dataset_name '/gbgs' '/' g_str],'file') mkdir([folder_name '/scanpaths/' dataset_name '/gbgs' '/' g_str]); end %folder
            if ~exist([folder_name '/bmaps/' dataset_name '/gbgs' '/' g_str],'file') mkdir([folder_name '/bmaps/' dataset_name '/gbgs' '/' g_str]); end %folder
            if ~exist([folder_name '/dmaps/' dataset_name '/gbgs' '/' g_str],'file') mkdir([folder_name '/dmaps/' dataset_name '/gbgs' '/' g_str]); end %folder
	    if ~exist([folder_name '/distmaps/' dataset_name '/gbgs' '/' g_str],'file') mkdir([folder_name '/distmaps/' dataset_name '/gbgs' '/' g_str]); end %folder
            scanpath = scanpath_comp_all_gbg_sep; save([folder_name '/scanpaths/' dataset_name '/gbgs/' g_str '/'  i_str '.mat'],'scanpath'); %scanpath
            imwrite(bmap_all_gbg_sep,[folder_name '/bmaps/' dataset_name '/gbgs/' g_str '/'  i_str '.png']);
            imwrite(dmap_all_gbg_sep,[folder_name '/dmaps/' dataset_name '/gbgs/' g_str '/'  i_str  '.png']);
	    imwrite(distmap_all_gbg_sep,[folder_name '/distmaps/' dataset_name '/gbgs/' g_str '/'  i_str  '.png']);
        end
    
end

