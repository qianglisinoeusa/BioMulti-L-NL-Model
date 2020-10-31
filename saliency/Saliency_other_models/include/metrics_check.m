cond_gotmetrics=[];
for idx=1:length(n_metrics_standard)
    mt=n_metrics_standard(idx);
    if isfield(metrics{mt},'score')
        cond_gotmetrics=[cond_gotmetrics,~isnan(metrics{mt}.score)]; %score is nan
        %cond_gotmetrics=[cond_gotmetrics,1];
    else
        cond_gotmetrics=[cond_gotmetrics,0]; %score not found
    end
end
% for idx=1:length(n_metrics_gazewise)
%     mt=n_metrics_gazewise(idx)
%     if isfield(metrics_gazewise{mt},'score')
%         cond_gotmetrics=[cond_gotmetrics,max(isnan(cell2mat(metrics_gazewise{mt}.score)))]; %score is nan
%         %cond_gotmetrics=[cond_gotmetrics,1];
%     else
%         cond_gotmetrics=[cond_gotmetrics,0]; %score not found
%     end
% end
% if sum(cond_gotmetrics)==0
%     %continue;
% end
%             