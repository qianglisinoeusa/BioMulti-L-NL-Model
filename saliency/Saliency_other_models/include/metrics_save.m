
            results_struct = struct();
            results_struct.metrics = metrics;
%             results_struct.metrics_pairwise = metrics_pairwise;
            results_struct.metrics_gazewise = metrics_gazewise;
            
            results_struct.indexes_all_trials = indexes_all_trials;
            results_struct.indexes_other_trials = indexes_other_trials;
                
            delete([output_folder '/' dataset '/' methods{m} '/' '*']);
            %rmdir([output_folder '/' dataset '/' methods{m}]);

            mkdir([output_folder '/' dataset '/' methods{m}]);
            save([output_folder '/' dataset '/' methods{m} '/' 'results.mat'], 'results_struct');
            