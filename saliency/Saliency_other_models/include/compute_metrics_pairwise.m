
                
            for pp=1:length(participants)
                params_folder_pairwise=params_folder;
                params_folder_pairwise.scanpaths_subfolder = [params_folder.scanpaths_subfolder '/pp/' num2str(pp)];
                params_folder_pairwise.bmaps_subfolder = [params_folder.bmaps_subfolder '/pp/' num2str(pp)];
                params_folder_pairwise.dmaps_subfolder = [params_folder.dmaps_subfolder '/pp/' num2str(pp)];
                params_folder_pairwise.baseline_subfolder = [params_folder.baseline_folder];
                params_folder_pairwise.fbaseline_subfolder = [params_folder.fbaseline_folder];
                params_folder_pairwise.mmaps_subfolder = [params_folder.mmaps_subfolder '/pp/' num2str(pp)];
                params_folder_pairwise.smaps_subfolder = [params_folder.smaps_subfolder ];
                params_folder_pairwise.scanpaths_predicted_subfolder = [params_folder.scanpaths_predicted_subfolder];
                
                disp(['participant=' num2str(pp)]);
                for idx=1:length(n_metrics_pairwise)
                    mt=n_metrics_pairwise(idx);
                    if find(n_missing_metrics_pairwise==mt)
                      disp(['Computing  ' metrics_pairwise{mt}.name '... ' ]);
                      [metrics_pairwise{mt}.score{pp},metrics_pairwise{mt}.sdev{pp},metrics_pairwise{mt}.score_all{pp},metrics_pairwise{mt}.roc_all{pp}] = feval(metrics_pairwise{mt}.type,metrics_pairwise{mt}.function, metrics_pairwise{mt}.baseline_type, metrics_pairwise{mt}.comparison_type,n_evaluations,metrics_pairwise{mt}.trials, metrics_pairwise{mt}.indexes_other, params_folder_pairwise);
                    else
                       %already calculated
                    end
                    
                    if strfind(metrics_pairwise{mt}.name,'AUC')>0 && ~isempty(find(n_missing_roc_pairwise==mt)) && isempty(find(n_missing_metrics_pairwise==mt)) %metric found but roc missing
                      disp(['Computing  ROC ' metrics_pairwise{mt}.name '... ' ]);
                      [~,~,~,metrics_pairwise{mt}.roc_all{pp}] = feval(metrics_pairwise{mt}.type,metrics_pairwise{mt}.function, metrics_pairwise{mt}.baseline_type, metrics_pairwise{mt}.comparison_type,n_evaluations,metrics_pairwise{mt}.trials, metrics_pairwise{mt}.indexes_other, params_folder_pairwise);
                    else
                       %already calculated
                    end
                    
                end
            end