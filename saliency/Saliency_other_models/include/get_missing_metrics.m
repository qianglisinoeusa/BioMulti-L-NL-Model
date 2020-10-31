n_missing_metrics=[];
n_missing_roc=[];
n_missing_metrics_pairwise=[];
n_missing_roc_pairwise=[];
n_missing_metrics_gazewise=[];
n_missing_metrics_gazewise_s=[];
n_missing_roc_gazewise=[];
n_missing_roc_gazewise_s=[];
if exist([output_folder '/' dataset '/' methods{m} '/' 'results.mat'], 'file')
    load([output_folder '/' dataset '/' methods{m} '/' 'results.mat']);
    %load old metrics
    if(isfield(results_struct,'metrics')) 
        for idx1=1:length(metrics_info)
            for idx2=1:length(results_struct.metrics)
                if isfield(results_struct.metrics{idx2},'name') 
                    if strcmp(metrics_info{idx1}.name,results_struct.metrics{idx2}.name) && length(find(n_metrics_standard==n_metrics_all(idx2)))>0
                        if isfield(results_struct.metrics{idx2},'score')
                            metrics{idx1}=results_struct.metrics{idx2};
                            n_missing_metrics=[n_missing_metrics, isnan(results_struct.metrics{idx2}.score).*idx2 ]; %score is nan
                        else
                            n_missing_metrics=[n_missing_metrics, n_metrics_all(idx2)];
                        end
                        if isfield(results_struct.metrics{idx2},'roc_all')
                            n_missing_roc=[n_missing_roc, isempty(results_struct.metrics{idx2}.roc_all).*idx2 ]; %roc is empty    
                        else
                            n_missing_roc=[n_missing_roc, n_metrics_all(idx2)];
                        end
                    end
                end
            end
        end
    else
        n_missing_metrics=n_metrics_standard;
        n_missing_roc=n_metrics_standard;
    end
%             if(isfield(results_struct,'metrics_pairwise')) 
%                 for idx1=1:length(metrics_pairwise_info)
%                     for idx2=1:length(results_struct.metrics_pairwise)
%                       if isfield(results_struct.metrics_pairwise{idx2},'name') 
%                           if strcmp(metrics_pairwise_info{idx1}.name,results_struct.metrics_pairwise{idx2}.name) && length(find(n_metrics_pairwise==n_metrics_all(idx2)))>0
%                               if isfield(results_struct.metrics_pairwise{idx2},'score')
%                                   metrics_pairwise{idx1}=results_struct.metrics_pairwise{idx2};
%                                   n_missing_metrics_pairwise=[n_missing_metrics_pairwise, max(isnan(cell2mat(results_struct.metrics_pairwise{idx2}.score))).*idx2]; %score is nan
%                               else
%                                   n_missing_metrics_pairwise=[n_missing_metrics_pairwise, n_metrics_all(idx2)];
%                               end
%                               if isfield(results_struct.metrics_pairwise{idx2},'roc_all')
%                                   n_missing_roc_pairwise=[n_missing_roc_pairwise, isempty(results_struct.metrics_pairwise{idx2}.roc_all{1}).*idx2 ]; %roc is empty    
%                               else
%                                   n_missing_roc_pairwise=[n_missing_roc_pairwise, n_metrics_all(idx2)];
%                               end
%                            end
%                        end
%                      end
%                  end
%             else
%                 n_missing_metrics_pairwise=n_metrics_pairwise;
%                 n_missing_roc_pairwise=n_metrics_pairwise;
%             end
    if(isfield(results_struct,'metrics_gazewise')) 
        for idx1=1:length(metrics_gazewise_info)
            for idx2=1:length(results_struct.metrics_gazewise)
                if isfield(results_struct.metrics_gazewise{idx2},'name') 
                    if strcmp(metrics_gazewise_info{idx1}.name,results_struct.metrics_gazewise{idx2}.name) && length(find(n_metrics_gazewise==n_metrics_all(idx2)))>0
                        if isfield(results_struct.metrics_gazewise{idx2},'score')
                            metrics_gazewise{idx1}=results_struct.metrics_gazewise{idx2};
                            try
                                n_missing_metrics_gazewise=[n_missing_metrics_gazewise, max(isnan(cell2mat(results_struct.metrics_gazewise{idx2}.score))).*idx2]; %score is nan
                            catch
                                n_missing_metrics_gazewise=[n_missing_metrics_gazewise, max(isnan(results_struct.metrics_gazewise{idx2}.score)).*idx2]; %score is nan
                            end
                        else
                            n_missing_metrics_gazewise=[n_missing_metrics_gazewise, n_metrics_all(idx2)];
                        end
                        if isfield(results_struct.metrics_gazewise{idx2},'roc_all')
                            try
                                n_missing_roc_gazewise=[n_missing_roc_gazewise, isempty(results_struct.metrics_gazewise{idx2}.roc_all{1}).*idx2 ]; %roc is empty    
                            catch
                                n_missing_roc_gazewise=[n_missing_roc_gazewise, n_metrics_all(idx2)];
                            end
                        else
                            n_missing_roc_gazewise=[n_missing_roc_gazewise, n_metrics_all(idx2)];
                        end
                        if isfield(results_struct.metrics_gazewise{idx2},'score_s')
                            metrics_gazewise{idx1}=results_struct.metrics_gazewise{idx2};
                            n_missing_metrics_gazewise_s=[n_missing_metrics_gazewise_s, max(isnan(cell2mat(results_struct.metrics_gazewise{idx2}.score_s))).*idx2]; %score is nan
                        else
                            n_missing_metrics_gazewise_s=[n_missing_metrics_gazewise_s, n_metrics_all(idx2)];
                        end
                        if isfield(results_struct.metrics_gazewise{idx2},'roc_all_s')
                            try
                                n_missing_roc_gazewise_s=[n_missing_roc_gazewise_s, isempty(results_struct.metrics_gazewise{idx2}.roc_all_s{1}).*idx2 ]; %roc is empty    
                            catch
                                n_missing_roc_gazewise_s=[n_missing_roc_gazewise_s, n_metrics_all(idx2)];
                            end
                        else
                            n_missing_roc_gazewise_s=[n_missing_roc_gazewise_s, n_metrics_all(idx2)];
                        end
                    end
                end
            end
        end
    else
        n_missing_metrics_gazewise=n_metrics_gazewise;
        n_missing_roc_gazewise=n_metrics_gazewise;
    end
else
    n_missing_metrics=n_metrics_standard;
    n_missing_metrics_pairwise=n_metrics_pairwise;
    n_missing_metrics_gazewise=n_metrics_gazewise;
    n_missing_metrics_gazewise_s=n_metrics_gazewise;
    n_missing_roc=n_metrics_standard;
    n_missing_roc_pairwise=n_metrics_pairwise;
    n_missing_roc_gazewise=n_metrics_gazewise;
end
%n_missing_metrics=[intersect(n_metrics_standard,n_metrics_mandatory)];
%n_missing_metrics_pairwise=[intersect(n_metrics_pairwise,n_metrics_mandatory)];
%n_missing_metrics_gazewise=[intersect(n_metrics_gazewise,n_metrics_mandatory)];
n_missing_metrics=[n_missing_metrics,zeros(1,length(n_metrics_standard)-length(n_missing_metrics))];
n_missing_metrics_pairwise=[n_missing_metrics_pairwise,zeros(1,length(n_metrics_pairwise)-length(n_missing_metrics_pairwise))];
n_missing_metrics_gazewise=[n_missing_metrics_gazewise,zeros(1,length(n_metrics_gazewise)-length(n_missing_metrics_gazewise))];
n_missing_metrics_gazewise_s=[n_missing_metrics_gazewise_s,zeros(1,length(n_metrics_gazewise)-length(n_missing_metrics_gazewise_s))];
n_missing_roc=[n_missing_roc,zeros(1,length(n_metrics_standard)-length(n_missing_roc))];
n_missing_roc_pairwise=[n_missing_roc_pairwise,zeros(1,length(n_metrics_pairwise)-length(n_missing_roc_pairwise))];
n_missing_roc_gazewise=[n_missing_roc_gazewise,zeros(1,length(n_metrics_gazewise)-length(n_missing_roc_gazewise))];

%force specific metrics (if exist, recompute and overwrite)
n_missing_metrics=[n_missing_metrics n_ow_metrics];

