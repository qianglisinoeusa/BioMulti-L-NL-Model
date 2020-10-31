function [ score_mean, score_std , score_all,roc_all, score_submetric] = metric_nonshuffled( metric ,baseline_type, comparison_type, n_evaluations, n_other_trials, indexes_other_trials, params_folder )

        
%         try
            score_all = NaN.*ones(1,n_evaluations);
            roc_all = num2cell(NaN.*ones(1,n_evaluations));
            score_sub_all=NaN.*ones(n_evaluations,20);
            indexes_single = 1:n_evaluations;
            for l=1:n_evaluations
                 try
                    %disp(['image: ' filenames_noext_cell{i} '#' int2str(i) '/' int2str(length(filenames_noext_cell))]);

                    k = indexes_single(l);

                    image= im2double(imread([params_folder.images_subfolder '/' params_folder.filenames_noext_cell{k} '.' params_folder.images_extension])); 

                    %GROUND TRUTH
                    if any(vertcat(strfind(comparison_type,'bmap')))
                        bmap = im2double(imread([params_folder.bmaps_subfolder '/' params_folder.filenames_noext_cell{k} '.' params_folder.bmap_extension])); 
                        clean_bmap(bmap, 0.9 );
                    end

                    if any(vertcat(strfind(comparison_type,'dmap')))
                        dmap = im2double(imread([params_folder.dmaps_subfolder '/' params_folder.filenames_noext_cell{k} '.' params_folder.dmap_extension])); 
                        dmap = resize_map(dmap,image); 
                        dmap = normalize_minmax(dmap);
                    end

                    if any(vertcat(strfind(comparison_type,'mmap')))
                        mmap = im2double(imread([params_folder.mmaps_subfolder '/' params_folder.filenames_noext_cell{k} '.' params_folder.mmap_extension])); 
                        mmap = normalize_minmax(mmap);
                        if size(mmap,3)>1, mmap=rgb2gray(mmap); end
                    end

                    if any(vertcat(strfind(comparison_type,'smap')))
                        smap = im2double(imread([params_folder.smaps_subfolder '/' params_folder.filenames_noext_cell{k} '.' params_folder.smap_extension])); 
                        smap = resize_map(smap,image); 
                        smap = normalize_minmax(smap);
                    end

                    if any(vertcat(strfind(comparison_type,'scanpath')))
                        scanpath = load([params_folder.scanpaths_subfolder '/' params_folder.filenames_noext_cell{k} '.' params_folder.scanpath_extension]); scanpath = scanpath.scanpath;
                        scanpath_predicted = load([params_folder.scanpaths_predicted_subfolder '/' params_folder.filenames_noext_cell{k} '.' params_folder.scanpath_extension]); scanpath_predicted = scanpath_predicted.scanpath;
                    end

                    if any(vertcat(strfind(comparison_type,'scanpath_saccades')))
                        if exist([params_folder.scanpaths_subfolder '/' 'saccades'],'file') 
                            params_folder.scanpaths_subfolder = [params_folder.scanpaths_subfolder '/' 'saccades'];
                        end
                        if exist([params_folder.scanpaths_predicted_subfolder '/' 'saccades'],'file') 
                            params_folder.scanpaths_predicted_subfolder = [params_folder.scanpaths_predicted_subfolder '/' 'saccades'];
                        end
                        scanpath = load([params_folder.scanpaths_subfolder '/' params_folder.filenames_noext_cell{k} '.' params_folder.scanpath_extension]); scanpath = scanpath.scanpath;
                        scanpath_predicted = load([params_folder.scanpaths_predicted_subfolder '/' params_folder.filenames_noext_cell{k} '.' params_folder.scanpath_extension]); scanpath_predicted = scanpath_predicted.scanpath;
                    end

                    switch comparison_type
                         case 'smap-bmap'
                            try
                                [score,tp,fp] = feval(metric,smap,bmap);
                            catch
                                [score] = feval(metric,smap,bmap);
                            end
                         case 'smap-mmap'
                            [score] = feval(metric,smap,mmap); %[score,~,score_sub]
                        case 'smap-dmap'
                            [score] = feval(metric,smap,dmap); %[score,~,score_sub]
                        case 'scanpath_single'
                            [score,~,score_sub] = feval(metric,scanpath_predicted); %[score,~,score_sub]
                        case 'scanpath_saccades_single'
                            [score,~,score_sub] = feval(metric,scanpath_predicted); %[score,~,score_sub]
                         case 'scanpath-mmap'
                            [score,score_sub] = feval(metric,scanpath_predicted,mmap); %[score,~,score_sub]
                         case 'scanpath-scanpath'
                            [score,~,score_sub] = feval(metric,scanpath_predicted,scanpath); %[score,~,score_sub]
                         case 'scanpath-pp_scanpath'
                            [score,~,score_sub] = feval(metric,scanpath_predicted,scanpath); %[score,~,score_sub]
                         case 'scanpath-gs_scanpath'
                             [tok1,tok2] = strtok(params_folder.scanpaths_subfolder,'gbg'); gaze = str2num(tok2(1,5:end));
                            [score,~,score_sub] = feval(metric,scanpath_predicted,scanpath,gaze); %[score,~,score_sub]
                         case 'scanpath-scanpath-scanpath_next-gaze'
                            [tok1,tok2] = strtok(params_folder.scanpaths_subfolder,'gbg'); gaze = str2num(tok2(1,5:end));
                            scanpaths_subfolder_next = [tok1 tok2(1,1:4) num2str(gaze+1)];
                            if exist([scanpaths_subfolder_next '/' params_folder.filenames_noext_cell{k} '.' params_folder.scanpath_extension],'file')
                                scanpath_next = load([scanpaths_subfolder_next '/' params_folder.filenames_noext_cell{k} '.' params_folder.scanpath_extension]); scanpath_next = scanpath_next.scanpath;
                                [score,~,score_sub] = feval(metric,scanpath_predicted,scanpath,scanpath_next,gaze); %[score,~,score_sub]
                            else
                                score = NaN;
                                score_sub=NaN;
                            end
                         case 'scanpath-scanpath_saccades'
                            [score,~,score_sub] = feval(metric,scanpath_predicted,scanpath); %[score,~,score_sub]
                          case 'scanpath_saccades-pp_scanpath'
                            [score,~,score_sub]= feval(metric,scanpath_predicted,scanpath); %[score,~,score_sub]
                         case 'scanpath_saccades-scanpath_saccades'
                            [score,~,score_sub]= feval(metric,scanpath_predicted,scanpath); %[score,~,score_sub]
                        case 'scanpath_saccades-gs_scanpath'
                            [tok1,tok2] = strtok(params_folder.scanpaths_subfolder,'gbg'); gaze = str2num(tok2(1,5:end));
                            [score,~,score_sub]= feval(metric,scanpath_predicted,scanpath,gaze); %[score,~,score_sub]
                         case 'scanpath-scanpath-gaze'
                            [tok1,tok2] = strtok(params_folder.scanpaths_subfolder,'gbg'); gaze = str2num(tok2(1,5:end));
                            [score,~,score_sub] = feval(metric,scanpath_predicted,scanpath,gaze); %[score,~,score_sub]
                         case 'scanpath-scanpath_saccades-scanpath_next-gaze'
                            [tok1,tok2] = strtok(params_folder.scanpaths_subfolder,'gbg'); gaze = str2num(tok2(1,5:end));
                            scanpaths_subfolder_next = [tok1 tok2(1,1:4) num2str(gaze+1)];
                            if exist([scanpaths_subfolder_next '/' params_folder.filenames_noext_cell{k} '.' params_folder.scanpath_extension],'file')
                                scanpath_next = load([scanpaths_subfolder_next '/' params_folder.filenames_noext_cell{k} '.' params_folder.scanpath_extension]); scanpath_next = scanpath_next.scanpath;
                               [score,~,score_sub]= feval(metric,scanpath_predicted,scanpath,scanpath_next,gaze); %[score,~,score_sub]
                            else
                                score = NaN;
                                score_sub=NaN;
                            end
                         case 'scanpath-scanpath_saccades-gaze'
                            [tok1,tok2] = strtok(params_folder.scanpaths_subfolder,'gbg'); gaze = str2num(tok2(1,5:end));
                            [score,~,score_sub]= feval(metric,scanpath_predicted,scanpath,gaze); %[score,~,score_sub]
                    end
                    score_all(l) = real(score); 
                    if exist('score_sub','var')
                        try
                        score_sub_all(l,1:length(score_sub))=score_sub;
                        catch
                        score_sub_all=score_sub;
                        end
                    else
                        score_sub_all(l,:)=NaN;
                    end
%                     if exist('tp') && exist('fp')
%                         roc_all=[roc_all; tp,fp];
%                     end
                    if exist('tp') && exist('fp')
                        roc_all{l}=[tp,fp];
                    end
                 catch
                     score_all(l)=NaN;
                 end
            end
            score_mean = nanmean(score_all);
            score_std = nanstd(score_all);
            score_submetric = nanmean(score_sub_all);

%             if ~isempty(roc_all)
%                 [tp_all,fp_all]=clean_roc(roc_all(:,1),roc_all(:,2));
%                 roc_all=[tp_all,fp_all];
%             end
%         catch 
%             score_mean = NaN;
%             score_std = NaN;
%         end
end
