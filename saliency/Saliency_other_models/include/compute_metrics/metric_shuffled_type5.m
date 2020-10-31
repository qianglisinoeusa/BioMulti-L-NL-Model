function [ score_shuffled, std_shuffled, score_all, roc_all,score_submetric] = metric_shuffled_type5( metric ,baseline_type, comparison_type, n_evaluations, n_other_trials, indexes_other_trials, params_folder )

        
%         try
            score_all = NaN.*ones(1,n_evaluations);
        roc_all = num2cell(NaN.*ones(1,n_evaluations));
    score_submetric=NaN;
            indexes_single = 1:n_evaluations;
            for l=1:n_evaluations
                %disp(['image: ' filenames_noext_cell{i} '#' int2str(i) '/' int2str(length(filenames_noext_cell))]);
                try 
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


                    if any(vertcat(strfind(comparison_type,'smap')))
                        smap = im2double(imread([params_folder.smaps_subfolder '/' params_folder.filenames_noext_cell{k} '.' params_folder.smap_extension])); 
                        smap = resize_map(smap,image); 
                        smap = normalize_minmax(smap);
                    end

                    if any(vertcat(strfind(comparison_type,'baseline')))
                        baseline_dmap = im2double(imread([params_folder.baseline_subfolder '/' 'baseline' '.' 'png'])); 
                        baseline_dmap = normalize_minmax(baseline_dmap);
                    end
                    if any(vertcat(strfind(comparison_type,'fbaseline')))
                        fbaseline_dmap = im2double(imread([params_folder.fbaseline_subfolder '/' 'fbaseline' '.' 'png'])); 
                        fbaseline_dmap = normalize_minmax(fbaseline_dmap);
                    end



                    switch comparison_type
                         case 'smap-bmap-baseline'
                            [score] = feval(metric,smap,bmap,baseline_dmap);
                        case 'smap-baseline'
                            [score] = feval(metric,smap,baseline_dmap);
                        case 'smap-fbaseline'
                            [score] = feval(metric,smap,fbaseline_dmap);
                    end
                    score_all(l) = real(score);
                catch 
                    score_all(l) = NaN;
                end
            end
            score_shuffled = nanmean(score_all);
            std_shuffled = nanstd(score_all);
%         catch 
%             score_shuffled = NaN;
%             std_shuffled = NaN;
%         end
end
