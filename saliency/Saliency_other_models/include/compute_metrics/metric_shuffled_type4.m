function [ score_shuffled, std_shuffled,score_all , roc_all,score_submetric] = metric_shuffled_type4( metric, baseline_type, comparison_type, n_evaluations,n_other_trials, indexes_other_trials,params_folder)
%     try
        score_all = NaN.*ones(1,n_evaluations);
        roc_all = num2cell(NaN.*ones(1,n_evaluations));
    score_submetric=NaN;
        for l=1:n_other_trials
            try
            %disp(['image: ' params_folder.filenames_noext_cell{l} '#' int2str(l) '/' int2str(length(params_folder.filenames_noext_cell))]);

            %GROUND TRUTH
            bmap = im2double(imread([bmaps_subfolder '/' params_folder.filenames_noext_cell{l} '.' params_folder.bmap_extension])); clean_bmap(bmap, 0.9 );
            smap = im2double(imread([smaps_subfolder '/' params_folder.filenames_noext_cell{l} '.' params_folder.smap_extension])); smap = normalize_minmax(resize_map(smap,bmap));
            dmap = im2double(imread([dmaps_subfolder '/' params_folder.filenames_noext_cell{l} '.' params_folder.dmap_extension])); dmap = normalize_minmax(resize_map(dmap,bmap));

            
            other_trials = indexes_other_trials(l,:);
            

            for i=1:n_evaluations
                k = other_trials(l);
                bmap_other = im2double(imread([bmaps_subfolder '/' params_folder.filenames_noext_cell{k} '.' params_folder.bmap_extension])); bmap_other = resize_map(bmap_other,bmap); clean_bmap(bmap_other, 0.9 ); 
                dmap_other = im2double(imread([dmaps_subfolder '/' params_folder.filenames_noext_cell{k} '.' params_folder.dmap_extension])); dmap_other = normalize_minmax(resize_map(dmap_other,dmap));

                
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
                score_trials(i) = real(score);
            end
            score_shuffled_trials = nanmean(score_trials);
            score_all(l) = score_shuffled_trials;
            catch
                score_all(l)=NaN;
            end
        end
        score_shuffled = nanmean(score_all);
        std_shuffled = nanstd(score_all);
%     catch 
%         score_shuffled = NaN;
%         std_shuffled = NaN;
%     end
end
