function [ models_list , models_list_function] = getmodels( models_path, models_prefix )

if nargin < 2, models_prefix='saliency_'; end
if nargin < 1, models_path='../modelos'; end


models_list={};
models_list_function={};
files = sort_nat(dirpath2listpath(dir(fullfile(models_path, []))));        %read files names
if size(files,1)>2
    models_list=files(3:end);
    for m=1:numel(models_list)
        models_list_function{m}=[models_prefix lower(models_list{m})];
    end
end

end

