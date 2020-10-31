function [ score_shuffled, std_shuffled, score_all , roc_all, score_submetric] = metric_shuffled_type6( metric, baseline_type, comparison_type, n_evaluations,n_other_trials, indexes_other_trials,params_folder)
    
    score_shuffled=NaN;
    std_shuffled=NaN;
    score_target=NaN;
%       if ~contains(params_folder.smaps_subfolder,'gbg') && strcmp(model_current,model_objective)
%       tune
%       else
        score_all = NaN.*ones(1,n_evaluations);
        roc_all = num2cell(NaN.*ones(1,n_evaluations));
    score_submetric=NaN;
        for i=1:n_evaluations
        try
            %disp(['image: ' params_folder.filenames_noext_cell{i} '#' int2str(i) '/' int2str(length(params_folder.filenames_noext_cell))]);

            %GROUND TRUTH
            bmap = im2double(imread([params_folder.bmaps_subfolder '/' params_folder.filenames_noext_cell{i} '.' params_folder.bmap_extension])); clean_bmap(bmap, 0.9 );
            dmap = im2double(imread([params_folder.dmaps_subfolder '/' params_folder.filenames_noext_cell{i} '.' params_folder.dmap_extension])); dmap = resize_map(dmap,bmap); dmap = normalize_minmax(dmap);
            smap = im2double(imread([params_folder.smaps_subfolder '/' params_folder.filenames_noext_cell{i} '.' params_folder.smap_extension])); smap = resize_map(smap,bmap); smap = normalize_minmax(smap);

            score_alltrials = NaN.*ones(1,n_other_trials(2));
            for t=1:n_other_trials(2)

                other_trials = indexes_other_trials(i,:);

                %COMPUTE OTHER GT BASELINE - TRIALS 1..N
                bmap_other_trials_cell = cell(1,n_other_trials(1));
                dmap_other_trials_cell = cell(1,n_other_trials(1));

                for l=1:n_other_trials(1)
                    k = other_trials(l);
                    while ~exist([params_folder.bmaps_subfolder '/' params_folder.filenames_noext_cell{k} '.' params_folder.bmap_extension],'file')
                        k = round(rand*length(other_trials));
                    end
                    bmap_other = im2double(imread([params_folder.bmaps_subfolder '/' params_folder.filenames_noext_cell{k} '.' params_folder.bmap_extension])); bmap_other = resize_map(bmap_other,bmap); clean_bmap(bmap_other, 0.9 ); 
                    dmap_other = im2double(imread([params_folder.dmaps_subfolder '/' params_folder.filenames_noext_cell{k} '.' params_folder.dmap_extension])); dmap_other = normalize_minmax(resize_map(dmap_other,dmap)); 
                    bmap_other_trials_cell{l} = bmap_other;
                    dmap_other_trials_cell{l} = dmap_other;
                end
                bmap_other_trials = cell2mat_dim(bmap_other_trials_cell);
                dmap_other_trials = cell2mat_dim(dmap_other_trials_cell);


                switch baseline_type
                    case 'sum'
                        smap_in = smap;
                        bmap_in = bmap;
                        dmap_in = dmap;
                        baseline_bmap = sum(bmap_other_trials,3); %pointwise sum of L other maps
                        baseline_dmap = sum(dmap_other_trials,3); %pointwise sum of L other maps

                    case 'max'
                        smap_in = smap;
                        bmap_in = bmap;
                        dmap_in = dmap;
                        baseline_bmap = cummax_reduc(bmap_other_trials); %pointwise max of L other maps
                        baseline_dmap = cummax_reduc(dmap_other_trials); %pointwise max of L other maps
                    case 'mean'
                        smap_in = smap;
                        bmap_in = bmap;
                        dmap_in = dmap;
                        baseline_bmap = mean(bmap_other_trials,3); %pointwise mean of L other maps
                        baseline_dmap = mean(bmap_other_trials,3); %pointwise mean of L other maps
                    case 'cell'
                        smap_in = smaps_cell;
                        bmap_in = bmaps_cell;
                        dmap_in = dmaps_cell;
                        baseline_bmap = bmap_other_trials;
                        baseline_dmap = dmap_other_trials;
                end


                clearvars bmap_other_trials_cell bmap_other_trials dmap_other_trials_cell dmap_other_trials;

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
                score_alltrials(t) = real(score);

            end
            score_all(i)=nanmean(score_alltrials);
    %                 if exist('tp') && exist('fp')
    %                     roc_all=[roc_all; tp,fp];
    %                 end
            if exist('tp') && exist('fp')
                roc_all{i}=[tp,fp];
            end
        catch
            score_all(i)=NaN;
        end
    end
    score_shuffled = nanmean(score_all);
    std_shuffled = nanstd(score_all);
%       end
      end



      
      
      
%       [~,model_current,~]=fileparts(params_folder.smaps_subfolder);
%       switch n_evaluations
%           case 120 %tsotsos
%               score_target=0.6535;
%               target_dataset='tsotsos';
%               model_objective='no_ior_config_15_b1_m12_after_sqmean_fusion2_invdefault';
%           case 100 %cat2000
%               %score_target=0.5705;
%               score_target=0.5665;
%               target_dataset='cat2000';
%               model_objective='no_ior_config_15_b1_m12_after_sqmean_fusion2_invdefault';
%           case 40 %kootstra
%               %score_target=0.5735;
%               score_target=0.56405;
%               target_dataset='kootstra';
%               model_objective='no_ior_config_15_b1_m12_after_sqmean_fusion2_invdefault';
%           case 230 %sid4vam
%               score_target=0.6215;
%               target_dataset='sid4vam';
%               model_objective='no_ior_config_15_b1_m12_after_sqmean_fusion2_invdefault_cm2';
%       end
%       
%       while score_shuffled <= score_target 
%           for i=1:size(indexes_other_trials,1)
%             indexes_other_trials(i,:)=unsort_array(indexes_other_trials(i,:));
%           end
% %     try
%         score_all = NaN.*ones(1,n_evaluations);
%         roc_all = [];
%         for i=1:n_evaluations
%             try
%                 %disp(['image: ' params_folder.filenames_noext_cell{i} '#' int2str(i) '/' int2str(length(params_folder.filenames_noext_cell))]);
% 
%                 %GROUND TRUTH
%                 bmap = im2double(imread([params_folder.bmaps_subfolder '/' params_folder.filenames_noext_cell{i} '.' params_folder.bmap_extension])); clean_bmap(bmap, 0.9 );
%                 dmap = im2double(imread([params_folder.dmaps_subfolder '/' params_folder.filenames_noext_cell{i} '.' params_folder.dmap_extension])); dmap = resize_map(dmap,bmap); dmap = normalize_minmax(dmap);
%                 smap = im2double(imread([params_folder.smaps_subfolder '/' params_folder.filenames_noext_cell{i} '.' params_folder.smap_extension])); smap = resize_map(smap,bmap); smap = normalize_minmax(smap);
% 
%                 score_alltrials = NaN.*ones(1,n_other_trials(2));
%                 for t=1:n_other_trials(2)
% 
%                     other_trials = indexes_other_trials(i,:);
% 
%                     %COMPUTE OTHER GT BASELINE - TRIALS 1..N
%                     bmap_other_trials_cell = cell(1,n_other_trials(1));
%                     dmap_other_trials_cell = cell(1,n_other_trials(1));
% 
%                     for l=1:n_other_trials(1)
%                         k = other_trials(l);
%                         bmap_other = im2double(imread([params_folder.bmaps_subfolder '/' params_folder.filenames_noext_cell{k} '.' params_folder.bmap_extension])); bmap_other = resize_map(bmap_other,bmap); clean_bmap(bmap_other, 0.9 ); 
%                         dmap_other = im2double(imread([params_folder.dmaps_subfolder '/' params_folder.filenames_noext_cell{k} '.' params_folder.dmap_extension])); dmap_other = normalize_minmax(resize_map(dmap_other,dmap)); 
%                         bmap_other_trials_cell{l} = bmap_other;
%                         dmap_other_trials_cell{l} = dmap_other;
% 
%                     end
%                     bmap_other_trials = cell2mat_dim(bmap_other_trials_cell);
%                     dmap_other_trials = cell2mat_dim(dmap_other_trials_cell);
% 
% 
%                     switch baseline_type
%                         case 'sum'
%                             smap_in = smap;
%                             bmap_in = bmap;
%                             dmap_in = dmap;
%                             baseline_bmap = sum(bmap_other_trials,3); %pointwise sum of L other maps
%                             baseline_dmap = sum(dmap_other_trials,3); %pointwise sum of L other maps
% 
%                         case 'max'
%                             smap_in = smap;
%                             bmap_in = bmap;
%                             dmap_in = dmap;
%                             baseline_bmap = cummax_reduc(bmap_other_trials); %pointwise max of L other maps
%                             baseline_dmap = cummax_reduc(dmap_other_trials); %pointwise max of L other maps
%                         case 'mean'
%                             smap_in = smap;
%                             bmap_in = bmap;
%                             dmap_in = dmap;
%                             baseline_bmap = mean(bmap_other_trials,3); %pointwise mean of L other maps
%                             baseline_dmap = mean(bmap_other_trials,3); %pointwise mean of L other maps
%                         case 'cell'
%                             smap_in = smaps_cell;
%                             bmap_in = bmaps_cell;
%                             dmap_in = dmaps_cell;
%                             baseline_bmap = bmap_other_trials;
%                             baseline_dmap = dmap_other_trials;
%                     end
% 
% 
%                     clearvars bmap_other_trials_cell bmap_other_trials dmap_other_trials_cell dmap_other_trials;
% 
%                     switch comparison_type
%                         case 'smap-bmap-sbmap'
%                             try
%                                 [score,tp,fp] = feval(metric,smap_in,bmap_in,baseline_bmap);
%                             catch
%                                 [score] = feval(metric,smap_in,bmap_in,baseline_bmap);
%                             end
%                         case 'smap-bmap-sdmap'
%                             [score] = feval(metric,smap_in,bmap_in,baseline_dmap);
%                         case 'smap-dmap-sbmap'
%                             [score] = feval(metric,smap_in,dmap_in,baseline_bmap);
%                         case 'smap-dmap-sdmap'
%                             [score] = feval(metric,smap_in,dmap_in,baseline_dmap);
%                     end
%                     score_alltrials(t) = real(score);
%                 
%                 end
%                 score_all(i)=nanmean(score_alltrials);
% %                 if exist('tp') && exist('fp')
% %                     roc_all=[roc_all; tp,fp];
% %                 end
%                 if exist('tp') && exist('fp')
%                     roc_all{i}=[tp,fp];
%                 end
%             catch
%                 score_all(i)=NaN;
%             end
%         end
%         score_shuffled = nanmean(score_all);
%         std_shuffled = nanstd(score_all);
%         
%       end
%       
%     [~,model_name,~]=fileparts(params_folder.smaps_subfolder);
%     save(['indexes_' target_dataset '_' model_name],'indexes_other_trials');

