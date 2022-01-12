%%
%%  MIT chromatic videos processing
%%  
%%  Decomposition chromatic video into luminance, RG, YB channels.
%% 
clear all; clc; close all;
%addpath(genpath('/media/disk/vista/Papers/A_Connectivity/n_dimensional_example/'));
%addpath(genpath('/media/disk/vista/Papers/A_Connectivity/'));
%addpath(genpath('/media/disk/vista/Papers/IMAGING/fMRI_explora/Vistalab'))
%addpath(genpath('/media/disk/vista/Papers/A_GermanyValencia/new_code_clean/BioMultiLayer_L_NL_color/colorlab/'))

%%
%% VideoReader plugin libmwgstreamerplugin failed to load properly. 
%% So I did next section in my local computer then saved L, RG, YB 
%% videos and transfer it to serve for 3D Fourier analyses.
%%
%%    Necessaries libraries to successfully launch VideoReader
%%    help docs: https://gstreamer.freedesktop.org/documentation/installing/on-linux.html?gi-language=c
%%    Install GStreamer on Ubuntu or Debian
%%
%%
%%    sudo add-apt-repository ppa:mc3man/gstffmpeg-keep
%%    sudo apt-get update
%%    sudo apt-get install gstreamer0.10-ffmpeg
%%    sudo apt install ubuntu-restricted-extras


VideoSet='/home/qiangli/Matlab/Stimuli/';

digitDatasetPath = fullfile(VideoSet);
Vs=dir([digitDatasetPath, '*.mp4']);
files={Vs.name};

%startcol
Mrgb2lms1 = [0.2973 0.6606 0.0773; 0.1061 0.6853 0.0757; 0.0118 0.0489 0.5638]; 
Mrgb2xyz = [0.4162    0.3148    0.2328;0.2311    0.6899    0.0790;0.0190    0.0786    0.9057];
Mxyz2lms = [0.2434    0.8524   -0.0516;-0.3954    1.1642    0.0837;    0         0    0.6225];
Mxyz2atd_jameson = [0 1 0;1 -1 0;0 0.4 -0.4];

Mrgb2james = Mxyz2atd_jameson*Mrgb2xyz;
Mrgb2lms = Mxyz2lms*Mrgb2xyz;  

for k=1:numel(files)
    k
    V= VideoReader(files{k});
    %video = read_video(files{k},0, 30);        
    L = VideoWriter([VideoSet,V.Name(1:end-4),'_L']);
    RG = VideoWriter([VideoSet,V.Name(1:end-4),'_RG']);
    YB = VideoWriter([VideoSet,V.Name(1:end-4),'_YB']);
    open(L);
    open(RG);
    open(YB);
    while(hasFrame(V))
        frame = double(readFrame(V));      
        ATDframe = transf_color_image(frame/255,Mrgb2james);
        AA=ATDframe(:,:,1);
        TT=ATDframe(:,:,2);
        DD=ATDframe(:,:,3);
          
        writeVideo(L, mat2gray(AA));
        writeVideo(RG,mat2gray(TT));
        writeVideo(YB,mat2gray(DD));
    end
    close(L);
    close(RG);
    close(YB);
end


%% 3D Fourier Transform for preprocessing lumiance videos to select subset videos
%% for postprocessing

          
