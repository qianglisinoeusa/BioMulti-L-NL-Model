

function [ ] = main( dataset, flag )
if nargin < 2, flag=-1; end

clearvars('-except','flag','dataset','filenames_db_name', 'bmaps_db_name' );

%recursive addpath
addpath(genpath('include'));

bmap_extension = 'png';
dmap_extension = 'png';
mmap_extension = 'png';
smap_extension = 'png';
scanpath_extension = 'mat';

input_folder = 'input';
output_folder = 'output';
images_folder = 'images';
filenames_folder = 'filenames';
smaps_folder = 'smaps';
dmaps_folder = 'dmaps'; 
fdmaps_folder = 'fdmaps';
mmaps_folder = 'mmaps';
bmaps_folder = 'bmaps';
baseline_folder = 'baseline'; 
fbaseline_folder = 'fbaseline'; 
scanpaths_folder = 'scanpaths';
%bmaps_db_folder = 'bmaps_db';
switch flag
    case 0
        gazes_num=0;
    case -1
        gazes_num=20;
    otherwise
        gazes_num=get_mediangazes([input_folder '/' bmaps_folder '/' dataset '/' 'gbg']);
end

%LOAD CELL WITH FILENAMES

extensions={'png','jpg','jpeg'};
images_extension='png';
for ext=1:length(extensions)
    filenames_cell = dirpath2listpath(dir([input_folder '/' images_folder '/' dataset '/' '*.' extensions{ext}])); 
    filenames_cell = sort_nat(filenames_cell);
    if (length(filenames_cell) > 0)
        images_extension = extensions{ext};
    end
end

filenames_db_name = '';
bmaps_db_name='';
% %filenames_db_name = 'sorted_tsotsos.mat';
% %bmaps_db_name = 'tsotsos_db.mat';
% %filenames_db_name = 'sorted_judd_768x1024.mat';
% %bmaps_db_name = 'judd_db.mat';
% %filenames_db_name = 'sorted_judd.mat';
% %bmaps_db_name = 'judd_db_full.mat';
% %filenames_db_name = 'sorted_cat2000.mat';
% %bmaps_db_name = 'cat2000_db.mat';


if exist([input_folder '/' filenames_folder '/' dataset '/' filenames_db_name]) == 0
    filenames_cell = dirpath2listpath(dir([input_folder '/' images_folder '/' dataset '/' '*.' images_extension])); filenames_cell = sort_nat(filenames_cell);
    %filenames_cell = dirpath2listpath(dir([input_folder '/' dmaps_folder '/' dataset '/' '*.' dmap_extension])); filenames_cell = sort_nat(filenames_cell);
    %filenames_cell = dirpath2listpath(dir([input_folder '/' bmaps_folder '/' dataset '/' '*.' bmap_extension])); filenames_cell = sort_nat(filenames_cell);
    filenames_noext_cell = filenames_cell;
    for f=1:length(filenames_cell)
        filenames_noext_cell{f} = remove_extension(filenames_cell{f});
    end
else
    load([input_folder '/' filenames_folder '/' dataset '/' filenames_db_name]);
    filenames_cell = FNAMESDB; 
    filenames_noext_cell = filenames_cell;
    for f=1:length(filenames_cell)
        filenames_noext_cell{f} = remove_extension(filenames_cell{f});
    end
end

%LOAD CELL WITH METHOD NAMES
methods = listpath_dir([input_folder '/' smaps_folder '/' dataset]);

rng('shuffle');
methods = unsort_cell(methods); %ables computing several threads without overlap


%LOAD CELL WITH SALIENCY METHODS, DISCARD INCOMPLETE METHODS
% smaps_cell = cell(length(methods),length(filenames_cell));

 n_methods = length(methods);
 m_back=0;
 current_folder=pwd;
if false
 for m=1:n_methods
     m=m-m_back;
     disp(['Checking ' methods{m} '...']);
     if m <= n_methods
         for i=1:length(filenames_cell)
             method_smaps_path=[input_folder '/' smaps_folder '/' dataset '/' methods{m}];
             if exist([method_smaps_path '/' filenames_noext_cell{i} '.' smap_extension],'file') % ...
                %&& exist([method_smaps_path '/' 'scanpath/' filenames_noext_cell{i} '.' scanpath_extension],'file') ...
                %&& exist([method_smaps_path '/' 'gbg/'],'file'))

                %smaps_cell{m,i} = imread([method_smaps_path '/' filenames_noext_cell{i} '.' smap_extension]);

                if ~exist([method_smaps_path '/' 'gbg/' int2str(gazes_num)],'file')
                    if exist([method_smaps_path '/' 'mean/'],'file') %gbg as mean
                        system(['rm -rf ' method_smaps_path '/' 'gbg']);
                        system(['ln -rsf ' method_smaps_path '/' 'mean/' ' ' method_smaps_path '/' 'gbg']);
                    end
                    if ~exist([method_smaps_path '/' 'gbg'],'file')
                        mkdir([method_smaps_path '/' 'gbg/']);
                    end
                    for g=1:gazes_num
                        if ~exist([method_smaps_path '/' 'gbg/' num2str(g)],'file')
                            system(['rm -rf ' method_smaps_path '/' 'gbg/' num2str(g)]);
                            system(['ln -rsf ' current_folder '/' method_smaps_path ' ' method_smaps_path '/' 'gbg/' num2str(g)]);
                        end
                    end
                end
                if ~exist([method_smaps_path '/' 'gbgs/' int2str(gazes_num)],'file')
                    if exist([method_smaps_path '/' 'gazes/'],'file') %gbg as mean
                        system(['rm -rf ' method_smaps_path '/' 'gbgs']);
                        system(['ln -rsf ' current_folder '/' method_smaps_path '/' 'gazes/' ' ' method_smaps_path '/' 'gbgs']);
                    end
                    if ~exist([method_smaps_path '/' 'gbgs'],'file')
                        mkdir([method_smaps_path '/' 'gbgs/']);
                    end
                    for g=1:gazes_num
                        if ~exist([method_smaps_path '/' 'gbgs/' num2str(g)],'file')
                            system(['rm -rf ' method_smaps_path '/' 'gbgs/' num2str(g)]);
                            system(['ln -rsf ' method_smaps_path ' ' method_smaps_path '/' 'gbgs/' num2str(g)]);
                        end
                    end
                end
             else
                 methods = remove_cell_data(methods,m);
                 n_methods = length(methods); %update methods
                 m_back=m_back+1;
                 break;
             end
             
         end
     end
 end
 
end

%LOAD CELL WITH DENSITY MAPS GT
%dmaps_cell = cell(1,length(filenames_cell));
% for i=1:length(filenames_cell)
%   dmaps_cell{i} = im2double(imread([input_folder '/' dmaps_folder '/' dataset '/' filenames_noext_cell{i} '.' dmap_extension]));
% end

%LOAD CELL WITH BINARY MAPS GT
% if exist([input_folder '/' bmaps_db_folder '/' dataset '/' bmaps_db_name]) ~= 0
%     load([input_folder '/' bmaps_db_folder '/' dataset '/' bmaps_db_name]);
%     bmaps_cell = PMAPSDB;
% else
%     bmaps_cell = cell(1,length(filenames_cell));
%     for i=1:length(filenames_cell)
%         bmaps_cell{i} = im2double(imread([input_folder '/' bmaps_folder '/' dataset '/' filenames_noext_cell{i} '.' bmap_extension]));
%     end
% end


clearvars('-except','dataset', 'gazes_num', 'filenames_noext_cell', 'input_folder', 'output_folder', 'methods', 'images_folder', 'images_extension','smaps_folder', 'smap_extension','dmaps_folder','fdmaps_folder', 'dmap_extension','bmaps_folder', 'bmap_extension','mmaps_folder', 'mmap_extension','baseline_folder','fbaseline_folder','scanpaths_folder','scanpath_extension','n_other_trials');

params_folder.images_folder=images_folder;
params_folder.images_extension=images_extension;
params_folder.smaps_folder=smaps_folder;
params_folder.smap_extension=smap_extension;
params_folder.dmaps_folder=dmaps_folder;
params_folder.dmap_extension=dmap_extension;
params_folder.fdmaps_folder=fdmaps_folder;
params_folder.bmaps_folder=bmaps_folder;
params_folder.bmap_extension=bmap_extension;
params_folder.mmaps_folder=mmaps_folder;
params_folder.mmap_extension=mmap_extension;
params_folder.baseline_folder=baseline_folder;
params_folder.fbaseline_folder=fbaseline_folder;
params_folder.scanpaths_folder=scanpaths_folder;
params_folder.scanpath_extension=scanpath_extension;
params_folder.filenames_noext_cell=filenames_noext_cell;
params_folder.gazes_num=gazes_num;

%COMPUTE METRICS (it saves results for each method on output)
get_metrics(dataset, filenames_noext_cell, input_folder, output_folder, methods,params_folder);


%SEE RESULTS (it reads results for each method on output and outputs csv)
see_results(dataset, output_folder,images_extension,filenames_noext_cell);






end

