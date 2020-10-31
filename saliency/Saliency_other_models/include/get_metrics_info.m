metrics = get_metrics_properties( n_trials1, indexes_other_trials);
metrics_info=metrics;

participants = 1:length(listpath_dir([params_folder.scanpaths_subfolder '/pp/']));
metrics_pairwise = metrics_info;
metrics_pairwise_info=metrics_info;
for idx=1:length(n_metrics_pairwise)
    mt=n_metrics_pairwise(idx);
    metrics_pairwise_info{mt}.name = [metrics_pairwise_info{mt}.name '_pairwise']; 
    metrics_pairwise{mt}.name = [metrics_pairwise{mt}.name '_pairwise'];
end

gazes = 1:length(listpath_dir([params_folder.scanpaths_subfolder '/gbg/']));
if length(gazes)>params_folder.gazes_num, gazes=1:params_folder.gazes_num; end
metrics_gazewise = metrics_info;
metrics_gazewise_info=metrics_info;
for idx=1:length(n_metrics_gazewise)
    mt=n_metrics_gazewise(idx);
    metrics_gazewise_info{mt}.name = [metrics_gazewise_info{mt}.name '_gazewise']; 
    metrics_gazewise{mt}.name = [metrics_gazewise{mt}.name '_gazewise']; 
end
        