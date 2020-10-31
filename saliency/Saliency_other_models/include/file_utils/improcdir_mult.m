
function [] = improcdir_mult(process,model,evaluation,evaluation_values,evaluation_conditions, mode, folder_props)



if nargin < 7
    folder_props.output_extension='.png';
    folder_props.output_folder=['output' ];
    folder_props.input_extension='.png';
    folder_props.input_folder=['input'];
    folder_props.mask_extension='.png';
    folder_props.mask_folder='masks';
    folder_props.smap_extension='.png';
    folder_props.smap_folder='smaps'; 
end

%input_folder = uigetdir(pwd, 'Select a folder');           %get folder dir

N_conditions=numel(evaluation_conditions);

    
switch mode
    case 0
            aux_input_folder=[folder_props.input_folder '/' evaluation];
            aux_output_folder=[folder_props.output_folder '/' model '/' evaluation];
            aux_smap_folder=[folder_props.smap_folder '/' model];
            aux_mask_folder=[folder_props.mask_folder];
            
            paths.input_image_paths=cell(1,N_conditions);
            paths.smap_image_paths=cell(1,N_conditions);
            paths.mask_image_paths=cell(1,N_conditions);
            paths.output_image_paths=cell(1,N_conditions);
            if N_conditions
                for c=1:N_conditions
                    folder_props.input_folder = [aux_input_folder '/' evaluation_conditions{c}];
                    folder_props.mask_folder = [folder_props.input_folder  '/' aux_mask_folder];
                    folder_props.smap_folder = [folder_props.input_folder  '/' aux_smap_folder];
                    folder_props.output_folder = [aux_output_folder]; 
                    
                    files = sort_nat(dirpath2listpath(dir(fullfile(folder_props.input_folder, ['*' folder_props.input_extension]))));
                    files_smap = sort_nat(dirpath2listpath(dir(fullfile(folder_props.smap_folder, ['*' folder_props.smap_extension]))));
                    files_mask = sort_nat(dirpath2listpath(dir(fullfile(folder_props.mask_folder, ['*' folder_props.mask_extension]))));

                    N_files = size(files,1);                                %readed number of files
                    N_smaps = size(files_smap,1); 
                    N_masks = size(files_mask,1); 
                    
                    paths.input_path=folder_props.input_folder;
                    paths.smap_path=folder_props.smap_folder;
                    paths.mask_path=folder_props.mask_folder;
                    paths.output_path=folder_props.output_folder;
                    if (N_files > 0 && N_files == N_smaps && N_files == N_masks && N_files == numel(evaluation_values))
                         paths.input_image_paths{c}=cell(1,N_files);
                         paths.smap_image_paths{c}=cell(1,N_files);
                         paths.mask_image_paths{c}=cell(1,N_files);
                         paths.output_image_paths{c}=cell(1,N_files);
                        for i=1:N_files
                            paths.input_image_paths{c}{i} = [folder_props.input_folder '/' files{i}];
                            paths.smap_image_paths{c}{i} = [folder_props.smap_folder '/' files_smap{i}];
                            paths.mask_image_paths{c}{i} = [folder_props.mask_folder '/' files_mask{i}];
                            paths.output_image_paths{c}{i} = [folder_props.output_folder '/' remove_extension(files{i}) '_' process folder_props.output_extension];      
                        end
                        mkdir(folder_props.output_folder);
                    end
		    
                end
            else
                
                
                folder_props.input_folder = [aux_input_folder ];
                folder_props.mask_folder = [folder_props.input_folder  '/' aux_mask_folder];
                folder_props.smap_folder = [folder_props.input_folder  '/' aux_smap_folder];
                folder_props.output_folder = [aux_output_folder]; 
                
                files = sort_nat(dirpath2listpath(dir(fullfile(folder_props.input_folder, ['*' folder_props.input_extension]))));
                files_smap = sort_nat(dirpath2listpath(dir(fullfile(folder_props.smap_folder, ['*' folder_props.smap_extension]))));
                files_mask = sort_nat(dirpath2listpath(dir(fullfile(folder_props.mask_folder, ['*' folder_props.mask_extension]))));

                N_files = size(files,1);                                %readed number of files
                N_smaps = size(files_smap,1); 
                N_masks = size(files_mask,1); 
                
                
                
                paths.input_path=folder_props.input_folder;
                paths.smap_path=folder_props.smap_folder;
                paths.mask_path=folder_props.mask_folder;
                paths.output_path=folder_props.output_folder;
                if (N_files > 0 && N_files == N_smaps && N_files == N_masks && N_files == numel(evaluation_values))
                     evaluation_conditions={evaluation};
                     paths.input_image_paths{1}=cell(1,N_files);
                     paths.smap_image_paths{1}=cell(1,N_files);
                     paths.mask_image_paths{1}=cell(1,N_files);
                     paths.output_image_paths{1}=cell(1,N_files);
                    for i=1:N_files
                        paths.input_image_paths{1}{i} = [folder_props.input_folder '/' files{i}];
                        paths.smap_image_paths{1}{i} = [folder_props.smap_folder '/' files_smap{i}];
                        paths.mask_image_paths{1}{i} = [folder_props.mask_folder '/' files_mask{i}];
                        paths.output_image_paths{1}{i} = [folder_props.output_folder '/' remove_extension(files{i}) '_' process folder_props.output_extension];      
                    end
		mkdir(folder_props.output_folder);
                end
		
            end
            
            if (N_files > 0 && N_files == N_smaps && N_files == N_masks && N_files == numel(evaluation_values))
                feval(process,model,evaluation, evaluation_values, evaluation_conditions,paths); 
            end
    otherwise
        ;
end



end
