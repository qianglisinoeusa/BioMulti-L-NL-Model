function [] = see_results(dataset,output_folder,images_extension, filenames_noext_cell)
addpath(genpath('include'));

if nargin < 2
    output_folder = 'output'; 
end;

if nargin<3
    images_extension='png';
end

if nargin<4
    filenames_cell = dirpath2listpath(dir(['input/images/' dataset '/' '*.' images_extension])); filenames_cell = sort_nat(filenames_cell);
    filenames_noext_cell = filenames_cell;
    for f=1:length(filenames_cell)
        filenames_noext_cell{f} = remove_extension(filenames_cell{f});
    end
end


selected_metrics; %n_metrics_all, n_metrics_standard, n_metrics_pairwise, n_metrics_gazewise
n_metrics = max(n_metrics_standard);

methods_all = listpath_dir([output_folder '/' dataset]);
SUPERMAT_scores = cell(length(methods_all)+1,length(n_metrics_standard)+1);
SUPERMAT_sdev = cell(length(methods_all)+1,length(n_metrics_standard)+1);

% pp_SUPERMAT_scores = cell(length(methods_all)+1,length(n_metrics_pairwise)+1);
% pp_SUPERMAT_sdev = cell(length(methods_all)+1,length(n_metrics_pairwise)+1);

g_SUPERMAT_scores = cell(length(methods_all)+1,length(n_metrics_gazewise)+1);
g_SUPERMAT_sdev = cell(length(methods_all)+1,length(n_metrics_gazewise)+1);

%% TODO
 i_SUPERMAT_scores = cell(length(n_metrics_all),1);
 i_SUPERMAT_names = cell(length(n_metrics_all),1);
 for n=1:length(n_metrics_all)
    i_SUPERMAT_scores{n}=cell(length(methods_all)+1,length(filenames_noext_cell)+1);
 end
 
 
for m=1:length(methods_all)
    
        
        
        
       %try
        results_method = load([output_folder '/' dataset '/' methods_all{m} '/' 'results.mat']); results_method = results_method.results_struct;
        
        methods_all{m}; %print
        
        %groupwise
        %SUPERMAT_scores{m} = cell(1,n_metrics+1);
        %SUPERMAT_sdev{m} = cell(1,n_metrics+1);

        SUPERMAT_scores{1,1} = 'method';
        SUPERMAT_scores{m+1,1} = methods_all{m}; %name
        SUPERMAT_sdev{1,1} = 'method';
        SUPERMAT_sdev{m+1,1} = methods_all{m}; %name
        
        
        for idx=1:length(n_metrics_standard)
            e = n_metrics_standard(idx);
            try
            if isfield(results_method.metrics{e},'score')
            SUPERMAT_scores{1,e+1} = results_method.metrics{e}.name;
            SUPERMAT_scores{m+1,e+1} = num2str(results_method.metrics{e}.score); if isempty(results_method.metrics{e}.score) SUPERMAT_scores{m+1,e+1} = 0; end
            SUPERMAT_sdev{1,e+1} = results_method.metrics{e}.name;
            SUPERMAT_sdev{m+1,e+1} = num2str(results_method.metrics{e}.sdev); if isempty(results_method.metrics{e}.sdev) SUPERMAT_sdev{m+1,e+1} = 0; end
            end
            end
        end

        %% TODO
        
        
        for idx=1:length(n_metrics_standard)
            e = n_metrics_standard(idx);
            try
            if isfield(results_method.metrics{e},'score')
            i_SUPERMAT_names{idx}=results_method.metrics{e}.name;
            
            i_SUPERMAT_scores{idx}{1,1} = 'method';
            i_SUPERMAT_scores{idx}{m+1,1} = methods_all{m}; %name
        
            for f=1:length(filenames_noext_cell)
                i_SUPERMAT_scores{idx}{1,f+1} = filenames_noext_cell{f};
                i_SUPERMAT_scores{idx}{m+1,f+1} = num2str(results_method.metrics{e}.score_all(f)); if isempty(results_method.metrics{e}.score_all(f)) i_SUPERMAT_scores{idx}{m+1,f+1} = 0; end
            end
            end
            end
        end


        
        %pairwise
        %pp_SUPERMAT_scores{m} = cell(1,n_metrics+1+(1*length(results_method.metrics.metrics_pairwise{1}.score)));
        %pp_SUPERMAT_sdev{m} = cell(1,n_metrics+1+(1*length(results_method.metrics.metrics_pairwise{1}.score)));
        
%         pp_SUPERMAT_scores{1,1} = 'method';
%         pp_SUPERMAT_scores{m+1,1} = methods_all{m}; %name
%         pp_SUPERMAT_sdev{1,1} = 'method';
%         pp_SUPERMAT_sdev{m+1,1} = methods_all{m}; %name
%         
%         prev = 1;
%         for idx=1:length(n_metrics_pairwise)
%             e = n_metrics_pairwise(idx);
%             try
%             if isfield(results_method.metrics_pairwise{e},'score')
%             for pp=1:length(results_method.metrics_pairwise{e}.score)
%                 pp_SUPERMAT_scores{1,(1*pp)+prev} = [results_method.metrics_pairwise{e}.name ' p' num2str(pp)]; 
%                 pp_SUPERMAT_scores{m+1,(1*pp)+prev} = num2str(results_method.metrics_pairwise{e}.score{pp}); %if isempty(results_method.metrics_pairwise{e}.score{pp}) pp_SUPERMAT_scores{m+1,e+1+(1*pp)} = 0; end
%                 pp_SUPERMAT_sdev{1,(1*pp)+prev} = [results_method.metrics_pairwise{e}.name ' p' num2str(pp)]; 
%                 pp_SUPERMAT_sdev{m+1,(1*pp)+prev} = num2str(results_method.metrics_pairwise{e}.sdev{pp}); %if isempty(results_method.metrics_pairwise{e}.sdev{pp}) pp_SUPERMAT_sdev{m+1,e+1+(1*pp)} = 0; end
%             end
%             prev = prev + length(results_method.metrics_pairwise{e}.score);
%             end
%             end
%         end
        
        %group gazewise
        %g_SUPERMAT_scores{m} = cell(1,n_metrics+1+(1*length(results_method.metrics.metrics_gazewise{1}.score)));
        %g_SUPERMAT_sdev{m} = cell(1,n_metrics+1+(1*length(results_method.metrics.metrics_gazewise{1}.score)));
        
        g_SUPERMAT_scores{1,1} = 'method';
        g_SUPERMAT_scores{m+1,1} = methods_all{m}; %name
        g_SUPERMAT_sdev{1,1} = 'method';
        g_SUPERMAT_sdev{m+1,1} = methods_all{m}; %name
        
        prev = 1;
        for idx=1:length(n_metrics_gazewise)
            e = n_metrics_gazewise(idx);
            try
            if isfield(results_method.metrics_gazewise{e},'score')
            for g=1:length(results_method.metrics_gazewise{e}.score)
                g_SUPERMAT_scores{1,(1*g)+prev} = [results_method.metrics_gazewise{e}.name ' g' num2str(g-1)];
                g_SUPERMAT_sdev{1,(1*g)+prev} = [results_method.metrics_gazewise{e}.name ' g' num2str(g)];
                try
                    g_SUPERMAT_scores{m+1,(1*g)+prev} = num2str(results_method.metrics_gazewise{e}.score{g}); %if isempty(g_SUPERMAT_scores{m+1,e+1+(1*g)}) g_SUPERMAT_scores{m+1,e+1+(1*g)} = 0; end
                    g_SUPERMAT_sdev{m+1,(1*g)+prev} = num2str(results_method.metrics_gazewise{e}.sdev{g}); %if isempty(g_SUPERMAT_sdev{m+1,e+1+(1*g)}) g_SUPERMAT_sdev{m+1,e+1+(1*g)} = 0; end
                catch
                    g_SUPERMAT_scores{m+1,(1*g)+prev} = num2str(results_method.metrics_gazewise{e}.score(g)); %if isempty(g_SUPERMAT_scores{m+1,e+1+(1*g)}) g_SUPERMAT_scores{m+1,e+1+(1*g)} = 0; end
                    g_SUPERMAT_sdev{m+1,(1*g)+prev} = num2str(results_method.metrics_gazewise{e}.sdev(g)); %if isempty(g_SUPERMAT_sdev{m+1,e+1+(1*g)}) g_SUPERMAT_sdev{m+1,e+1+(1*g)} = 0; end
                end
            end
            prev = prev + length(results_method.metrics_gazewise{e}.score);
            end
            end
        end
        
	%catch
	%	continue;
	%end

	
        
end

SUPERMAT_scores; %print

w_csv(SUPERMAT_scores,[output_folder '/' dataset '_results_all.csv']);
w_csv(SUPERMAT_sdev,[output_folder '/' dataset '_results_all_sdev.csv']);

for n=1:length(i_SUPERMAT_names)
    i_SUPERMAT_names{n}; %print
    i_SUPERMAT_scores{n}; %print
    w_csv(i_SUPERMAT_scores{n},[output_folder '/' dataset '_results_all_' i_SUPERMAT_names{n} '.csv']);
end

% pp_SUPERMAT_scores
% 
% w_csv(pp_SUPERMAT_scores,[output_folder '/' dataset '_pp_results_all.csv']);
% w_csv(pp_SUPERMAT_sdev,[output_folder '/' dataset '_pp_results_all_sdev.csv']);

g_SUPERMAT_scores; %print

w_csv(g_SUPERMAT_scores,[output_folder '/' dataset '_g_results_all.csv']);
w_csv(g_SUPERMAT_sdev,[output_folder '/' dataset '_g_results_all_sdev.csv']);









end
