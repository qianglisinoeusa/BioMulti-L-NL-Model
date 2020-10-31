function [ paths ] = getpaths( models_list , models_list_function, evaluation_list, evaluation_conditions, evaluation_values_str, evaluation_values)
    
    for e=1:numel(evaluation_list)
        evaluation_conditions{e}
        for c=1:numel(evaluation_conditions{e})
            for v=1:numel(evaluation_values_str{e})
                paths.input_image_paths{e}{c}{v}=[pwd '/input/' evaluation_list{e} '/' evaluation_conditions{e}{c} '/' evaluation_values_str{e}{v}];
				paths.input_image_paths_dataset{e}{c}{v}=[pwd '/dataset/' getname_dataset(evaluation_list{e},evaluation_conditions{e}{c},evaluation_values_str{e}{v},'png')];
                
				paths.mask_image_paths{e}{c}{v}=[pwd '/input/' evaluation_list{e} '/' evaluation_conditions{e}{c} '/' 'masks/' evaluation_values_str{e}{v}];
				paths.mask_image_paths_dataset{e}{c}{v}=[pwd '/dataset/' 'masks/' getname_dataset(evaluation_list{e},evaluation_conditions{e}{c},evaluation_values_str{e}{v},'png')];
				
                paths.smap_image_paths{e}{c}{v}=cell(size(models_list,1),1);
                paths.output_image_paths{e}{c}{v}=cell(size(models_list,1),1);
                
                for m=1:numel(models_list)
                    paths.smap_image_paths{e}{c}{v}{m}=[pwd '/input/' evaluation_list{e} '/' evaluation_conditions{e}{c} '/' 'smaps/' models_list{m} '/' evaluation_values_str{e}{v}];
					paths.smap_image_paths_dataset{e}{c}{v}{m}=[pwd '/dataset/' 'smaps/' models_list{m} '/' getname_dataset(evaluation_list{e},evaluation_conditions{e}{c},evaluation_values_str{e}{v},'png')];
					
                    paths.results_qualitative_paths{e}{c}{v}{m}=[pwd '/output/' evaluation_list{e} '/' models_list{m}  '/' evaluation_conditions{e}{c} '/' 'qualitative_' evaluation_values_str{e}{v}];
                    paths.results_quantitative_paths{e}{v}{m}=[pwd '/output/' evaluation_list{e} '/' models_list{m}  '/' 'quantitative_' evaluation_values_str{e}{v}];
                    paths.results_quantitative_csv_paths{e}{m}=[pwd '/output/' evaluation_list{e} '/' models_list{m} '/' 'results.csv'];
                    paths.results_quantitative_lineplot_paths{e}{m}=[pwd '/output/' evaluation_list{e} '/' models_list{m}  '/'  'results.png'];
                end
            end
        end
        paths.results_all_csv_paths{e}=['output/' evaluation_list{e} '/' 'results.csv'];
        paths.results_all_lineplot_paths{e}=['output/' evaluation_list{e} '/' 'results.png'];
    end
    paths.input_path='input/';
    paths.output_path='output/';
end
