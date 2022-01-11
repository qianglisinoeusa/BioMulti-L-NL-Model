%%
%%  MIT chromatic videos processing
%%  
%%  Decomposition chromatic video into luminance, RG, YB channels.
%% 
clear all; clc; close all;

VideoSet='/home/liqiang/Downloads/AlgonautsVideos268_All_30fpsmax/';

digitDatasetPath = fullfile(VideoSet);
Vs=dir([digitDatasetPath, '*.mp4']);
files={Vs.name};

%startcol
%cd /home/liqiang/Downloads/AlgonautsVideos268_All_30fpsmax/
To = [35 0 0]; 
Mrgb2lms1 = [0.2973 0.6606 0.0773; 0.1061 0.6853 0.0757; 0.0118 0.0489 0.5638]; 
Mrgb2xyz = [0.4162    0.3148    0.2328;0.2311    0.6899    0.0790;0.0190    0.0786    0.9057];
Mxyz2lms = [0.2434    0.8524   -0.0516;-0.3954    1.1642    0.0837;    0         0    0.6225];
Mxyz2atd_jameson = [0 1 0;1 -1 0;0 0.4 -0.4];

Mrgb2james = Mxyz2atd_jameson*Mrgb2xyz;
Mrgb2lms = Mxyz2lms*Mrgb2xyz;  

for k=1:numel(files)
    k
    V= VideoReader(files{k});
    L = VideoWriter([VideoSet,V.Name(1:end-4),'_L','.mp4']);
    RG = VideoWriter([VideoSet,V.Name(1:end-4),'_RG','.mp4']);
    YB = VideoWriter([VideoSet,V.Name(1:end-4),'_YB','.mp4']);
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
