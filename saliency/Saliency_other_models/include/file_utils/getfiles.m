function [evaluation_list, evaluation_conditions, evaluation_values_str, evaluation_values] = getfiles(  )



evaluation_list={};
evaluation_conditions={};
evaluation_values={};

files = sort_nat(dirpath2listpath(dir(fullfile('input', []))));        %read files names
if size(files,1)>2
    evaluation_list=files(3:end);
    for e=1:numel(evaluation_list)
        files = sort_nat(dirpath2listpath(dir(fullfile(['input/' evaluation_list{e}], []))));        %read files names
        if size(files,1)>2
            evaluation_conditions{e}=files(3:end);
            for c=1:numel(evaluation_conditions{e})
                files = sort_nat(dirpath2listpath(dir(fullfile(['input/' evaluation_list{e} '/' evaluation_conditions{e}{c}], ['*.png']))));        %read files names
                if size(files,1)<1				
					files = sort_nat(dirpath2listpath(dir(fullfile(['input/' evaluation_list{e} '/' evaluation_conditions{e}{c}], ['*.jpg']))));        %read files names
                end
                if size(files,1)>1
                    evaluation_values_str{e}=files(1:end); 
                    for v=1:numel(evaluation_values_str{e}), evaluation_values{e}{v}=str2num(remove_extension(evaluation_values_str{e}{v})); end
                end
            end
        end
    end
end

end

