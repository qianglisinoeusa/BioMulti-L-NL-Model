%% This code used for gathering other saliency model result with MIT1003 database

%% QiangLI, Valencia, Spain

%% Read MIT1003 database
clear all;
clc;

addpath(genpath('models/'));
%addpath(genpath('export_fig-master/'));

path1 = 'download_parse_datasets/sid4vam/Achanta/';
path2 = 'download_parse_datasets/sid4vam/AIM/';
path3 = 'download_parse_datasets/sid4vam/HFT/';
path4 = 'download_parse_datasets/sid4vam/ICL/';
path5 = 'download_parse_datasets/sid4vam/SIM/';
path_itt = '~/QiangLi/Python_Utils_Functional/FixaTons/SID4VAM/saliency-master/saliency-master/input/smaps/toronto/ITT/' 

srcFiles = dir('/home/qiang/QiangLi/Python_Utils_Functional/FixaTons/SID4VAM/saliency-master/saliency-master/input/images/toronto/*.png');  

for i = 1:120
    i
    filename = strcat('/home/qiang/QiangLi/Python_Utils_Functional/FixaTons/SID4VAM/saliency-master/saliency-master/input/images/toronto/', srcFiles(i).name);
    [filpath, names, ext] = fileparts(filename);
    I = imread(filename);
    gbvs_install
    % get saliency map:
    smap = ittikochmap(filename); 
    smap=smap.master_map;
    name = strcat(path_itt, names, ext) ;
    imwrite(smap,name)  

    % Achanta model
    %smap_Achanta = saliency_achanta(I);
    %name = strcat(path1, names, ext) ;
    %imwrite(smap_Achanta,name)  
    
    % AIM model   TAKES LONG TIME !!!
    %smap_AIMs = AIM(filename,1); 
    %smap_AIM = uint8(smap_AIMs);
    %name = strcat(path2, names, ext) ;
    %imwrite(smap_AIM,name)  
    
    % HFT
    %input_image=double(I)./255;
    %[rows, columns, numberOfChannels] = size(input_image);
    %if numberOfChannels < 2
    %    input_image = repmat(input_image, [1,1,3]);
    %end
    %smap_hft = HFT(input_image);
    %name = strcat(path3, names, ext) ;
    %imwrite(smap_hft,name)  
    

    % ICL
    %smap_ICL = ICL(filename);
    %name = strcat(path4, names, ext) ;
    %imwrite(smap_ICL, name)  
    
    
    %SIM
    %[rows, columns, numberOfChannels] = size(I);
    %if numberOfChannels < 2
    %    I = repmat(I, [1,1,3]);
    %end
    %[m, n, p]      = size(I);
    %window_sizes = [13 26];                       
    %wlev         = min([7,floor(log2(min([m n])))]);
    %gamma        = 2.4;                             
    %srgb_flag    = 1;                               
    %smap_SIM = SIM(I, window_sizes, wlev, gamma, srgb_flag);
    %smapSIM = uint8(smap_SIM);
    %name = strcat(path5, names, ext) ;
    %imwrite(smapSIM, name)  
    
end