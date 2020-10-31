
function [] = get_metrics(dataset, filenames_noext_cell, input_folder, output_folder, methods, params_folder )
%dataset = dataset name [string]
%filenames_noex_cell: image filenames cell [i*string, without extension]
%input_folder: input of GT and smaps
%output_folder: output of results
%methods: saliency methods names [m*string, cell of strings]
%params_folder: struct of folder and file paths
%     smaps_folder: saliency maps (prediction) [string, 'smaps']
%     smap_extension: either 'png','jpg','jpeg'...
%     dmaps_folder: fixation density maps (GT) [string, 'dmaps']
%     dmap_extension: either 'png','jpg','jpeg'...
%     bmaps_folder: fixation binary maps (GT) [string, 'bmaps']
%     bmap_extension: either 'png','jpg','jpeg'...
%     mmaps_folder: window mask maps (GT) [string, 'bmaps']
%     mmap_extension: either 'png','jpg','jpeg'...
%     scanpaths_folder: fixation raw data (GT) [string, 'scanpaths']
%     scanpath_extension: to be read, 'mat'...
  

     %delete(gcp);
     %parpool('local',2);
     
     %% GET METRICS
     for m=1:length(methods) %saliency method %    parfor m=1:length(methods) %saliency method
        
        %% GET FOLDER LISTS
        params_folder.scanpaths_subfolder = [input_folder '/' params_folder.scanpaths_folder '/' dataset ];
        params_folder.images_subfolder = [input_folder '/' params_folder.images_folder '/' dataset ];
        params_folder.bmaps_subfolder = [input_folder '/' params_folder.bmaps_folder '/' dataset ];
        params_folder.dmaps_subfolder = [input_folder '/' params_folder.dmaps_folder '/' dataset ];
        params_folder.baseline_subfolder = [input_folder '/' params_folder.baseline_folder '/' dataset ];
        params_folder.fbaseline_subfolder = [input_folder '/' params_folder.fbaseline_folder '/' dataset ];
        params_folder.mmaps_subfolder = [input_folder '/' params_folder.mmaps_folder '/' dataset ];
        params_folder.smaps_subfolder = [input_folder '/' params_folder.smaps_folder '/' dataset '/' methods{m}];
        params_folder.scanpaths_predicted_subfolder = [params_folder.smaps_subfolder '/scanpath']; 
        
        %% SELECT METRICS
        selected_metrics; %n_metrics_all, n_metrics_standard, n_metrics_pairwise, n_metrics_gazewise

        %% SELECT TRIALS (MAX OF 100 for shuffled metrics)
        get_trials_idx;
        
        %% GET METRICS DESCRIPTIONS
        get_metrics_info;
        
        %% SEE IF METRICS WERE ALREADY CALCULATED
        get_missing_metrics;
        
        %% COMPUTE METRICS
            disp(['method: ' methods{m} '#' int2str(m) '/' int2str(length(methods))]);
            
            %% COMPUTE METRICS GROUPWISE
            compute_metrics_groupwise;
             
%             %% COMPUTE METRICS PAIRWISE
%             compute_metrics_pairwise;
            
            %% COMPUTE METRICS GAZEWISE
%             compute_metrics_gazewise_single;
            compute_metrics_gazewise;
            
            %% DO NOT SAVE RESULTS IF THERE IS NAN FOR ALL METRICS
            metrics_check;
            
            %% SAVE RESULT FILES
            metrics_save;
        end
    


end

                


