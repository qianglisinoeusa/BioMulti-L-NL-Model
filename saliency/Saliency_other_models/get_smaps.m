function [  ] = get_smaps( models_folder, datasets_list  )

    addpath(genpath('include'));    
    
    input_folder=[pwd '/' 'input'];
    images_folder=[pwd '/' 'input/images'];
    smaps_folder=[pwd '/' 'input/smaps'];  
    
    if nargin<2
    	datasets_list=listpath_dir(images_folder);
    end
    
    if nargin<1
        models_folder='../modelos';
    end
    [ models_list , models_list_function] = getmodels( models_folder );

    aux_PATH=getenv('PATH');
    current_path=pwd;
   
    for d=1:numel(datasets_list)
        dataset_name=datasets_list{d};
        images_list=listpath([images_folder '/' dataset_name]);
        for m=1:numel(models_list_function)
            model_name=models_list{m}
            model_path=[models_folder '/' model_name];
            addpath(genpath(model_path)); 
            addpath(genpath([current_path '/include']));
            %setenv('PATH',model_path);
            cd([pwd '/' model_path])
            mkdir([smaps_folder '/' dataset_name '/' model_name]);
             try 
                for i=1:numel(images_list)
                    image_name=images_list{i}; 
                    image_name_noext=remove_extension(image_name);
                    image_path=[images_folder '/' dataset_name '/' image_name];

                    smap_name=[image_name_noext '.png'];
                    smap_path=[smaps_folder '/' dataset_name '/' model_name '/' smap_name];

                    scanpath_name=[image_name_noext '.mat'];
                    scanpath_path=[smaps_folder '/' dataset_name '/' model_name '/' 'scanpath'];

                    smaps_single_path=[smaps_folder '/' dataset_name '/' model_name '/' 'gbgs'];
                    smaps_mix_path=[smaps_folder '/' dataset_name '/' model_name '/' 'gbg'];

                    switch nargout(models_list_function{m})
                        case 2
                            if ~exist(smap_path,'file') || ~exist([scanpath_path '/' scanpath_name],'file')
                                [smap,scanpath]=saliency_model(models_list_function{m},image_path);
                                imwrite(uint8(smap),smap_path);

                                mkdir(scanpath_path);
                                save([scanpath_path '/' scanpath_name],'scanpath');
                            end
                        case 3
                            if ~exist(smap_path,'file') || ~exist([scanpath_path '/' scanpath_name],'file') || ~exist(smaps_mix_path,'file') || ~exist(smaps_single_path,'file')
                                [smap,scanpath,smaps]=saliency_model(models_list_function{m},image_path);
                                imwrite(mat2gray(smap),smap_path);

                                mkdir(scanpath_path);
                                save([scanpath_path '/' scanpath_name],'scanpath');

                                mkdir(smaps_single_path);
                                for g=1:size(smaps,3)
                                    mkdir([smaps_single_path '/' num2str(g)]);
                                    smap_single=smaps(:,:,g);
                                    imwrite(mat2gray(smap_single),[smaps_single_path '/' num2str(g) '/' smap_name]);
                                end

                                mkdir(smaps_mix_path);
                                for g=1:size(smaps,3)
                                    mkdir([smaps_mix_path '/' num2str(g)]);
                                    smap_mix=mix_smaps(smaps,'mean',g);
                                    imwrite(smap_mix,[smaps_mix_path '/' num2str(g) '/' smap_name]);
                                end
                            end
                        otherwise
                            if ~exist(smap_path,'file')
                                [smap]=saliency_model(models_list_function{m},image_path);
                                imwrite(uint8(smap),smap_path);
                            end
                    end
                end
             catch exc_model_error
                disp(['error on model: ' model_name]);
                disp(getReport(exc_model_error,'extended'));
             end
            cd(current_path);
            %setenv('PATH',current_path);
        end
    end
    


end


	
