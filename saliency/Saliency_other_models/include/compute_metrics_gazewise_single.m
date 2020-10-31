
for g=1:length(gazes)

    params_folder_gazewise=params_folder;
%                 params_folder_gazewise.scanpaths_subfolder = [params_folder.scanpaths_subfolder '/gbg/' num2str(g)];
    params_folder_gazewise.scanpaths_subfolder = [params_folder.scanpaths_subfolder];
    params_folder_gazewise.bmaps_subfolder = [params_folder.bmaps_subfolder '/gbgs/' num2str(g)];
    params_folder_gazewise.dmaps_subfolder = [params_folder.dmaps_subfolder '/gbgs/' num2str(g)];
    params_folder_gazewise.baseline_subfolder = [params_folder.baseline_subfolder];
    params_folder_gazewise.mmaps_subfolder = [params_folder.mmaps_subfolder];
    params_folder_gazewise.smaps_subfolder = [params_folder.smaps_subfolder '/gbgs/' num2str(g) ];
    params_folder_gazewise.scanpaths_predicted_subfolder = [params_folder.scanpaths_predicted_subfolder];

    %non-scanpath models -> link to static saliency maps
    if ~exist([params_folder.smaps_subfolder '/gbgs/' ],'file')
        system(['mkdir ' params_folder.smaps_subfolder '/gbgs/']);
    end
    if ~exist(params_folder_gazewise.smaps_subfolder,'file')
        system(['ln -s ' params_folder.smaps_subfolder ' ' params_folder_gazewise.smaps_subfolder]);
    end

    disp(['gaze=' num2str(g)]);

    for idx=1:length(n_metrics_gazewise)
        mt=n_metrics_gazewise(idx);
        %if exist(params_folder_gazewise.scanpaths_predicted_subfolder,'file')

            if find(n_missing_metrics_gazewise_s==mt)
                disp(['Computing  ' metrics_gazewise{mt}.name '... ' ]);
                [metrics_gazewise{mt}.score_s{g},metrics_gazewise{mt}.sdev_s{g},metrics_gazewise{mt}.score_all_s{g},metrics_gazewise{mt}.roc_all_s{g}] = feval(metrics_gazewise{mt}.type,metrics_gazewise{mt}.function, metrics_gazewise{mt}.baseline_type, metrics_gazewise{mt}.comparison_type,n_evaluations,metrics_gazewise{mt}.trials, metrics_gazewise{mt}.indexes_other,params_folder_gazewise);
            else
                %already calculated
            end
            if ~isempty(strfind(metrics_gazewise{mt}.name,'AUC')) && ~isempty(find(n_missing_roc_gazewise_s==mt)) && isempty(find(n_missing_metrics_gazewise_s==mt)) %metric found but roc missing
                disp(['Computing  ROC ' metrics_gazewise{mt}.name '... ' ]);
                [~,~,~,metrics_gazewise{mt}.roc_all_s{g}] = feval(metrics_gazewise{mt}.type,metrics_gazewise{mt}.function, metrics_gazewise{mt}.baseline_type, metrics_gazewise{mt}.comparison_type,n_evaluations,metrics_gazewise{mt}.trials, metrics_gazewise{mt}.indexes_other,params_folder_gazewise);
            else
                %already calculated
            end
        %end
    end
end


