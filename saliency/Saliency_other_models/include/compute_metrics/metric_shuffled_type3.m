function [ score_shuffled, std_shuffled ,score_all,roc_all,score_submetric] = metric_shuffled_type3( metric, baseline_type, comparison_type,n_evaluations,n_other_trials, indexes_other_trials,params_folder)
%     try
        score_all = NaN.*ones(1,n_evaluations);
        roc_all = num2cell(NaN.*ones(1,n_evaluations));
    score_submetric=NaN;
        for l=1:n_other_trials
            try
            %disp(['image: ' params_folder.filenames_noext_cell{l} '#' int2str(l) '/' int2str(length(params_folder.filenames_noext_cell))]);
            
            nother_trials = indexes_other_trials(l,:);
            smaps_cell = cell(1,n_evaluations);
            bmaps_cell = cell(1,n_evaluations);
            dmaps_cell = cell(1,n_evaluations);
            
            %COMPUTE OTHER GT BASELINE - TRIALS 1..N
            bmap_other_trials_cell = cell(1,n_evaluations);
            dmap_other_trials_cell = cell(1,n_evaluations);
            
            for i=1:n_evaluations
                bmap = im2double(imread([params_folder.bmaps_subfolder '/' params_folder.filenames_noext_cell{i} '.' params_folder.bmap_extension])); clean_bmap(bmap, 0.9 );
                dmap = im2double(imread([params_folder.dmaps_subfolder '/' params_folder.filenames_noext_cell{i} '.' params_folder.dmap_extension])); dmap = normalize_minmax(resize_map(dmap,bmap));
                smap = im2double(imread([params_folder.smaps_subfolder '/' params_folder.filenames_noext_cell{i} '.' params_folder.smap_extension])); smap = normalize_minmax(resize_map(smap,bmap));

                
                smaps_cell{i} = smap;
                dmaps_cell{i} = dmap;
                
                k = other_trials(i);
                
                bmap_other = im2double(imread([params_folder.bmaps_subfolder '/' params_folder.filenames_noext_cell{k} '.' params_folder.bmap_extension])); bmap_other = resize_map(bmap_other,bmap); clean_bmap(bmap_other, 0.9 ); 
                dmap_other = im2double(imread([params_folder.dmaps_subfolder '/' params_folder.filenames_noext_cell{k} '.' params_folder.dmap_extension])); dmap_other = normalize_minmax(resize_map(dmap_other,bmap));


                bmap_other_trials_cell{i} = bmap_other;
                dmap_other_trials_cell{i} = dmap_other;
                
                
            end
            bmap_other_trials = cell2mat_dim(bmap_other_trials_cell);
            dmap_other_trials = cell2mat_dim(dmap_other_trials_cell);
            
            switch baseline_type
                case 'cell'
                    smap_in = smaps_cell;
                    bmap_in = bmaps_cell;
                    dmap_in = dmaps_cell;
                    baseline_bmap = bmap_other_trials;
                    baseline_dmap = dmap_other_trials;
            end


            switch comparison_type
                case 'smap-bmap-sbmap'
                    try
                        [score,tp,fp] = feval(metric,smap_in,bmap_in,baseline_bmap);
                    catch
                        [score] = feval(metric,smap_in,bmap_in,baseline_bmap);
                    end
                case 'smap-bmap-sdmap'
                    [score] = feval(metric,smap_in,bmap_in,baseline_dmap);
                case 'smap-dmap-sbmap'
                    [score] = feval(metric,smap_in,dmap_in,baseline_bmap);
                case 'smap-dmap-sdmap'
                    [score] = feval(metric,smap_in,dmap_in,baseline_dmap);


            end
            score_all(l) = real(score);
%             if exist('tp') && exist('fp')
%                 roc_all=[roc_all; tp,fp];
%             end
            if exist('tp') && exist('fp')
                roc_all{l}=[tp,fp];
            end
            catch
                score_all(l)=NaN;
            end
        end
        score_shuffled = nanmean(score_all);
        std_shuffled = nanstd(score_all);
        
%         if ~isempty(roc_all)
%             [tp_all,fp_all]=clean_roc(roc_all(:,1),roc_all(:,2));
%             roc_all=[tp_all,fp_all];
%         end
%     catch 
%         score_shuffled = NaN;
%         std_shuffled = NaN;
%     end
end
