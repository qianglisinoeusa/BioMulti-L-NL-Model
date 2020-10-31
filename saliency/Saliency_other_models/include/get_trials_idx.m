    n_evaluations = length(filenames_noext_cell);
            n_other_trials = n_evaluations-1;
            n_trials1 = 99;
            n_trials2 = 10;
            if n_trials1 > n_other_trials
                n_trials1 = n_other_trials;
            end
            if n_trials2 > n_other_trials
                n_trials2 = n_other_trials;
            end
            indexes_all_trials = 1:n_evaluations;
            indexes_other_trials = zeros(n_evaluations,n_other_trials);
            indexes_allother_trials = zeros(n_evaluations,n_evaluations-1);
            indexes_nother_trials = zeros(n_evaluations,n_evaluations); 
            
            sizes = zeros(n_evaluations,2);
            for i=1:n_evaluations
               k = indexes_all_trials(i);
               image = im2double(imread([params_folder.images_subfolder '/' params_folder.filenames_noext_cell{k} '.' params_folder.images_extension])); 
               sizes(i,:) = [size(image,1) size(image,2)];     
            end
        %% GET TRIALS AND OTHER IDX FOR SHUFFLED METRICS
            for i=1:n_evaluations
                        %k = indexes_all_trials(i);
                        %bmap = im2double(imread([bmaps_subfolder '/' filenames_noext_cell{k} '.' bmap_extension])); clean_bmap(bmap, 0.9 );
                        %rand_other_trials= idx_randperm_samesize(n_evaluations, n_other_trials, size(bmap), sizes, i); 
                        %all_other_trials= idx_allperm_samesize( n_evaluations, size(bmap), sizes, i);
                    rand_other_trials= idx_randperm(n_evaluations,n_other_trials,i);
                    all_other_trials= idx_allperm(n_evaluations,i);
                    indexes_other_trials(i,:) = rand_other_trials;
                    indexes_allother_trials(i,:) = all_other_trials;
            end
            
            for l=1:n_other_trials
                    rindex = randi(n_evaluations); %to determine each with same size as rindex (could be 1)
                        %k = indexes_all_trials(rindex);
                        %bmap = im2double(imread([bmaps_subfolder '/' filenames_noext_cell{k} '.' bmap_extension])); clean_bmap(bmap, 0.9 );
                        %rand_nother_trials = idx_randperm_samesize(n_evaluations, n_evaluations, size(bmap), sizes, rindex);  
                    rand_nother_trials = idx_randperm(n_evaluations, n_evaluations, rindex);  
                    indexes_nother_trials(l,:) = rand_nother_trials;
            end
            