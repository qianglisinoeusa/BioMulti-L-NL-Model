function [ score_shuffled, std_shuffled,score_all,roc_all ,score_submetric] = metric_shuffled_type2( metric, baseline_type, comparison_type, n_evaluations,n_other_trials, indexes_other_trials,params_folder )
%     try
        score_all = NaN.*ones(1,n_evaluations);
        roc_all = num2cell(NaN.*ones(1,n_evaluations));
    score_submetric=NaN;
        for i=1:n_evaluations
            try
            %disp(['image: ' params_folder.filenames_noext_cell{i} '#' int2str(i) '/' int2str(length(params_folder.filenames_noext_cell))]);

            %GROUND TRUTH
            bmap = im2double(imread([params_folder.bmaps_subfolder '/' params_folder.filenames_noext_cell{i} '.' params_folder.bmap_extension])); clean_bmap(bmap, 0.9 );
            dmap = im2double(imread([params_folder.dmaps_subfolder '/' params_folder.filenames_noext_cell{i} '.' params_folder.dmap_extension])); dmap = normalize_minmax(resize_map(dmap,bmap));
            smap = im2double(imread([params_folder.smaps_subfolder '/' params_folder.filenames_noext_cell{i} '.' params_folder.smap_extension])); smap = normalize_minmax(resize_map(smap,bmap));
            other_trials = indexes_other_trials(i,:);
            

            for l=1:n_other_trials
                k = other_trials(l);
                bmap_other = im2double(imread([params_folder.bmaps_subfolder '/' params_folder.filenames_noext_cell{k} '.' params_folder.bmap_extension])); bmap_other = resize_map(bmap_other,bmap); clean_bmap(bmap_other, 0.9 ); 
                dmap_other = im2double(imread([params_folder.dmaps_subfolder '/' params_folder.filenames_noext_cell{k} '.' params_folder.dmap_extension])); dmap_other = normalize_minmax(resize_map(dmap_other,dmap));
                
                score_trials = zeros(1,n_other_trials);
                switch comparison_type
                    case 'smap-bmap-sbmap'
                        [score] = feval(metric,smap,bmap,bmap_other);
                    case 'smap-bmap-sdmap'
                        [score] = feval(metric,smap,bmap,dmap_other);
                    case 'smap-dmap-sbmap'
                        [score] = feval(metric,smap,dmap,bmap_other);
                    case 'smap-dmap-sdmap'
                        [score] = feval(metric,smap,dmap,dmap_other);
                end
                score_trials(l) = real(score);
            end
            score_shuffled_trials = nanmean(score_trials);
            score_all(i) = score_shuffled_trials;
            catch
                score_all(i) = NaN;
            end
        end
        score_shuffled = nanmean(score_all);
        std_shuffled = nanstd(score_all);
%     catch 
%         score_shuffled = NaN;
%         std_shuffled = NaN;
%     end
end
