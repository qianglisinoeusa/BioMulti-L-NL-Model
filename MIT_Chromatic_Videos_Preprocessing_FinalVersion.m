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

%%()
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


VideoSet='/home/qiangli/Matlab/MIT_Video/Stimuli/';
cd /home/qiangli/Matlab/MIT_Video/Stimuli/;

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

%% Color selection for postprocessing
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; clc; close all;
VideoSet='/home/qiangli/Matlab/MIT_Video/Stimuli/';
VideoSet_c='/home/qiangli/Matlab/MIT_Video/Stimuli_Color/';
cd /home/qiangli/Matlab/MIT_Video/Stimuli/;
digitDatasetPath = fullfile(VideoSet);
Vs=dir([digitDatasetPath, '*.mp4']);
files_L={Vs.name};
load MIT_Chromatic_Video_Saturation_God

%% Not good

% SA1 = [];
% SA2 = [];
% SA3 = [];
% for i=1:numel(files_L) 
%     i
%     colors_in_video = [];
%     average_color = [];
%     V= VideoReader(files_L{i});
%     while(hasFrame(V))
%         frame = double(readFrame(V));
%         colors = double(reshape(frame,[prod(size(frame(:,:,1))) 3]));
%         average_col = mean(colors);
%         
%         colors_in_video = [colors_in_video;colors(1:5:end,:)];
%         average_color = [average_color;average_col];
%                 
%         %HSVframe = rgb2hsv(frame);
%         %Satur1 = squeeze(  mean(HSVframe, [1, 3])  );  %let average H,S,V channel
% 
%         %Satur2 = mean(mean(HSVframe(:,:,2)));  %let average H,S,V channel
% 
%     end
%     average_color = mean(average_color);
%     lab = rgb2lab(colors_in_video/255,'WhitePoint',average_color/255);
%     
%     %Satur3 = mean(  sqrt( lab(:,2).^2+lab(:,3).^2 )./lab(:,1) );
% 
%     Satur3 = mean(  sqrt( lab(:,2).^2+lab(:,3).^2 ) );
% 
%     
%     %SA1 = [SA1 mean(Satur1)];
%     %SA2 = [SA2 mean(Satur2)];
%     SA3 = [SA3 mean(Satur3)];
% 
% end
% save ('MIT_Chromatic_Video_Saturation.mat', 'SA3', '-v7.3');
save_c = 1;
SA4 = [];
mean_cols = [];
[mat,idx4] = sortrows(SA4');
for i=1:numel(files_L) 
    i
    [SA,mean_color] = chrom_or_achrom(VideoSet, files_L{i}); 
    SA4 = [SA4 SA];
    mean_cols = [mean_cols;mean_color];
    
    if save_c ==1
        if  SA>= median(mat)  
            source = fullfile(VideoSet,files_L{i});
            destination = fullfile(VideoSet_c,'More_Saturation/');
            copyfile(source,destination)
        else
            source = fullfile(VideoSet,files_L{i});
            destination = fullfile(VideoSet_c,'Less_Saturation/');
            copyfile(source,destination)

        end
    end
    
end

save ('MIT_Chromatic_Video_Saturation_God.mat', 'SA4','mean_cols', '-v7.3');
%load MIT_Chromatic_Video_Satureation
%load MIT_Chromatic_Video_Saturation
load MIT_Chromatic_Video_Saturation_God

%SA=sort(SA,'ascend');
%x = 1:size(SA,2); y = SA; 
%scatter(x, SA, 'linewidth',3); %scatter(x,y); 
%a = [1:size(SA,2)]'; b = num2str(a); c = cellstr(b);
%dx = 0.1; dy = 0.1; % displacement so the text does not overlay the data points
%text(x+dx, y+dy, c, 'Fontsize', 5);
%set(gcf, 'color', [1 1 1])

%figure(23),plot(sort(SA,'ascend'), 'linewidth',2), xlabel('Num of videos'), ylabel('Average of Saturation'), box off;
%set(gcf, 'color', [1 1 1])

% [mat,idx1] = sortrows(SA1');
% x = 1:size(SA1,2); y = mat; 
% figure(), scatter(x, y, 'linewidth',2); 
% b = num2str(idx1); c = cellstr(b);
% dx = 0.1; dy = 0.1; % displacement so the text does not overlay the data points
% text(x+dx, y+dy, c, 'Fontsize', 5);
% xlabel('Num of videos'), ylabel('Average of Saturation'), box off;
% set(gcf, 'color', [1 1 1])
% 
% [mat,idx2] = sortrows(SA2');
% x = 1:size(SA2,2); y = mat; 
% figure(),scatter(x, y, 'linewidth',2); 
% b = num2str(idx2); c = cellstr(b);
% dx = 0.1; dy = 0.1; % displacement so the text does not overlay the data points
% text(x+dx, y+dy, c, 'Fontsize', 5);
% xlabel('Num of videos'), ylabel('Average of Saturation'), box off;
% set(gcf, 'color', [1 1 1])
% 
% 
% [mat,idx3] = sortrows(SA3');
% x = 1:size(SA3,2); y = mat; 
% figure(),scatter(x, y, 'linewidth',2); 
% b = num2str(idx3); c = cellstr(b);
% dx = 0.1; dy = 0.1; % displacement so the text does not overlay the data points
% text(x+dx, y+dy, c, 'Fontsize', 5);
% xlabel('Num of videos'), ylabel('Average of Saturation'), box off;
% set(gcf, 'color', [1 1 1])

[mat,idx4] = sortrows(SA4');
x = 1:size(SA4,2); y = mat; 
figure(),scatter(x, y, 'linewidth',2); 
b = num2str(idx4); c = cellstr(b);
dx = 0.1; dy = 0.1; % displacement so the text does not overlay the data points
text(x+dx, y+dy, c, 'Fontsize', 5);
hold on,
plot(x,repmat(median(mat), 1102,1),'linewidth',2)
text(1102, median(mat), 'cutoff', 'Fontsize', 15);
xlabel('Num of videos'), ylabel('Saturation'), box off;
set(gcf, 'color', [1 1 1])

%thres_sat1 = max(mat)/2;
thres_sat2 = median(mat);


%% 3D Fourier Transform for preprocessing lumiance videos to select subset motion-related videos for postprocessing


clear all; clc; close all;

VideoSet='/home/qiangli/Matlab/MIT_Video/Stimuli/';
VideoSet_s = '/home/qiangli/Matlab/MIT_Video/Stimuli_Motion/';
VideoSet_t = '/home/qiangli/Matlab/MIT_Video/Stimuli_Texture/';
cd /home/qiangli/Matlab/MIT_Video/Stimuli/;
digitDatasetPath = fullfile(VideoSet);
Vs=dir([digitDatasetPath, '*.mp4']);
files_L={Vs.name};
load MIT_Chromatic_Video_SpeedEn_God

[mat5,~] = sortrows(speed_curve');
thres_speed2 = median(mat5);

[mat6,~] = sortrows(texture_curve');
thres_texture2 = median(mat6);

save_v=1;
save_t=2;
speed_curve=[];
texture_curve=[];


[x,y,t,fx,fy,ft] = spatio_temp_freq_domain(268,268,30,64,64,30);

% Conditions texture
f_cut_off1 = 0.2;
f_cut_off2 = 5;
f_cut_off3 = 30;

low_freq_residual = (fx.^2 + fy.^2) < f_cut_off1^2;
low_freq = ((fx.^2 + fy.^2) >= f_cut_off1^2) & ((fx.^2 + fy.^2) < f_cut_off2^2);
high_freq = ((fx.^2 + fy.^2) >= f_cut_off2^2) & ((fx.^2 + fy.^2) < f_cut_off3^2);

% Conditions motion

fs = sqrt(fx.^2 + fy.^2);
v_cut_off = 0.1;

low_speed = ( abs(ft) < abs(fs*v_cut_off) ) & ( (fx.^2 + fy.^2) > f_cut_off1^2  );
high_speed = ( abs(ft) >= abs(fs*v_cut_off) ) & ( (fx.^2 + fy.^2) > f_cut_off1^2  );


for i=1:numel(files_L) 
    i
    V= read_video_no_plot(files_L{i}, 0, 30);
    V = squeeze(V(:,:,2,:));
    %now2then vs then2now
    
    Y = now2then(V);
    FTY = fft3(Y , 1);
    
    Y_low = FTY.*low_freq;
    Y_high = FTY.*high_freq;
    
    hf_spatial_text = sum(abs(Y_high(:)).^2)/sum(abs(Y_low(:)).^2);

    Y_low_s = FTY.*low_speed;
    Y_high_s = FTY.*high_speed;
    
    hs_movie = sum(abs(Y_high_s(:)).^2)/sum(abs(Y_low_s(:)).^2);
    
    texture_curve = [texture_curve hf_spatial_text];
    speed_curve = [speed_curve hs_movie];

    
  
    %% motion   
    if save_v==1
        if hs_movie >= thres_speed2  
            source = fullfile(VideoSet,files_L{i});
            destination = fullfile(VideoSet_s,'high/');
            copyfile(source,destination)
        else
            source = fullfile(VideoSet,files_L{i});
            destination = fullfile(VideoSet_s,'low/');
            copyfile(source,destination)

        end
    end
    
    %% texture
    if save_t==2
        if hf_spatial_text >= thres_texture2  
            source = fullfile(VideoSet,files_L{i});
            destination = fullfile(VideoSet_t,'high/');
            copyfile(source,destination)
        else
            source = fullfile(VideoSet,files_L{i});
            destination = fullfile(VideoSet_t,'low/');
            copyfile(source,destination)

        end
    end

end


save('MIT_Chromatic_Video_SpeedEn_God.mat', 'speed_curve', 'texture_curve', '-v7.3');
load MIT_Chromatic_Video_SpeedEn_God

[mat5,idx5] = sortrows(speed_curve');
thres_speed1 = max(mat5)/2;
thres_speed2 = median(mat5);

x =   1:size(speed_curve,2); y = mat5; 
figure(),scatter(x, y, 'linewidth',2); 
b = num2str(idx5); c = cellstr(b);
dx = 0.1; dy = 0.1; % displacement so the text does not overlay the data points
text(x+dx, y+dy, c, 'Fontsize', 5);
hold on,
plot(x,repmat(thres_speed2, 1102,1),'linewidth',2)
text(1102, thres_speed2, 'cutoff', 'Fontsize', 15);
xlabel('Num of videos'), ylabel('Speed energy'), box off;
set(gcf, 'color', [1 1 1])

[mat6,idx6] = sortrows(texture_curve');
x =   1:size(texture_curve,2); y = mat6; 
thres_texture1 = max(mat6)/2;
thres_texture2 = median(mat6);

figure(),scatter(x, y, 'linewidth',2); 
b = num2str(idx6); c = cellstr(b);
dx = 0.1; dy = 0.1; % displacement so the text does not overlay the data points
text(x+dx, y+dy, c, 'Fontsize', 5);
hold on,
plot(x,repmat(thres_texture2, 1102,1),'linewidth',2)
text(1102, thres_texture2, 'cutoff', 'Fontsize', 15);
xlabel('Num of videos'), ylabel('Texture energy'), box off;
set(gcf, 'color', [1 1 1])


