 for idx=1:length(n_metrics_standard)
    mt=n_metrics_standard(idx);
    if find(n_missing_metrics==mt)
        disp(['Computing  ' metrics{mt}.name '... ' ]);
        [metrics{mt}.score,metrics{mt}.sdev,metrics{mt}.score_all,metrics{mt}.roc_all,metrics{mt}.score_submetric] = feval(metrics{mt}.type,metrics{mt}.function, metrics{mt}.baseline_type, metrics{mt}.comparison_type,n_evaluations,metrics{mt}.trials, metrics{mt}.indexes_other,params_folder);
        
        if ~isempty(strfind(metrics{mt}.comparison_type,'scanpath'))
            metrics_gazewise{mt}.score=metrics{mt}.score_submetric;
        end
    else
        %already calculated 
    end
    if ~isempty(strfind(metrics{mt}.name,'AUC')) && ~isempty(find(n_missing_roc==mt)) && isempty(find(n_missing_metrics==mt)) %metric found but roc missing
        disp(['Computing  ROC ' metrics{mt}.name '... ' ]);
        [~,~,~,metrics{mt}.roc_all,~] = feval(metrics{mt}.type,metrics{mt}.function, metrics{mt}.baseline_type, metrics{mt}.comparison_type,n_evaluations,metrics{mt}.trials, metrics{mt}.indexes_other,params_folder);
    else
        %already calculated 
    end
 end
