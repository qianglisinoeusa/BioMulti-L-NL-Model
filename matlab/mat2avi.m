%% The code used convert mat to avi for traning deep neural networks.(Now we have total 160 clips for 4 classes, maybe we need more for training)

%% QiangLI, Valencia, Spain
clear; clc; 
for small_clip =1:160
    
    %% clear video writing 
    Vedio_clear = VideoWriter(['/media/disk/databases/BBDD_video_image/statist_pelis_youtube/orson/data_video_achrom/clip_',num2str(small_clip),'_clear_image_25_frames.avi']);
    load (['/media/disk/databases/BBDD_video_image/statist_pelis_youtube/orson/data_video_achrom/clip_',num2str(small_clip),'_full_image_25_frames'],'v');
    Vedio_clear.open;
    for frame = 1:25
        frame_layer_clear = v(1:end, 1:end, frame);
        frame_layer_clear = mat2gray(frame_layer_clear); 
        Vedio_clear.writeVideo(frame_layer_clear);
    end
    Vedio_clear.close;
   
   %% noise video writing
   Vedio_noise = VideoWriter(['/media/disk/databases/BBDD_video_image/statist_pelis_youtube/orson/data_video_achrom/clip_',num2str(small_clip),'_noise_image_25_frames.avi']);
   load (['/media/disk/databases/BBDD_video_image/statist_pelis_youtube/orson/data_video_achrom/clip_',num2str(small_clip),'_full_image_25_frames'],'vn');
   Vedio_noise.open;
   for frame = 1:25
        frame_layer_nosie = vn(1:end, 1:end, frame);
        frame_layer_nosie = mat2gray(frame_layer_nosie); 
        Vedio_noise.writeVideo(frame_layer_nosie);
   end
   Vedio_noise.close;
   
   %% blur video writing 
   Vedio_blur = VideoWriter(['/media/disk/databases/BBDD_video_image/statist_pelis_youtube/orson/data_video_achrom/clip_',num2str(small_clip),'_blur_image_25_frames.avi']);
   load (['/media/disk/databases/BBDD_video_image/statist_pelis_youtube/orson/data_video_achrom/clip_',num2str(small_clip),'_full_image_25_frames'],'vb');
   Vedio_blur.open;
   for frame = 1:25
        frame_layer_blur = vb(1:end, 1:end, frame);
        frame_layer_blur  = mat2gray(frame_layer_blur); 
        Vedio_blur.writeVideo(frame_layer_blur);
   end
   Vedio_blur.close;
   
   %% blur-noise video writing 
   Vedio_blur_noise = VideoWriter(['/media/disk/databases/BBDD_video_image/statist_pelis_youtube/orson/data_video_achrom/clip_',num2str(small_clip),'_blur_noise_image_25_frames.avi']);
   load (['/media/disk/databases/BBDD_video_image/statist_pelis_youtube/orson/data_video_achrom/clip_',num2str(small_clip),'_full_image_25_frames'],'vbn');
   Vedio_blur_noise.open;
   for frame = 1:25
        frame_layer_blur_noise = vbn(1:end, 1:end, frame);
        frame_layer_blur_noise = mat2gray(frame_layer_blur_noise); 
        Vedio_blur_noise.writeVideo(frame_layer_blur_noise);
   end
   Vedio_blur_noise.close;
   
   clear v vb vbn vn;
end

cd /media/disk/databases/BBDD_video_image/statist_pelis_youtube/orson/data_video_achrom/

!mkdir clear_clips
!mkdir noise_clips
!mkdir blur_clips
!mkdir noise_blur_clips

!mv *_clear_image_25_frames.avi  clear_clips
!mv *_noise_image_25_frames.avi  noise_clips
!mv *_blur_image_25_frames.avi  blur_clips
!mv *_blur_noise_image_25_frames.avi  noise_blur_clips
!mv noise_clips/*_blur_noise_image_25_frames.avi noise_blur_clips

!mkdir video_avi_datasets
!mv clear_clips  video_avi_datasets
!mv noise_clips  video_avi_datasets
!mv blur_clips  video_avi_datasets
!mv noise_blur_clips video_avi_datasets


%% clear video writing 
for small_clip =1:9
    
    Vedio_clear = VideoWriter(['/media/disk/vista/Papers/A_Frontiers_CSF/extra_python_code/results/train_JoV/Videos_of_RestoreNet_Reuslt/out_video_',num2str(small_clip),'.avi']);
    load (['/media/disk/vista/Papers/A_Frontiers_CSF/extra_python_code/results/train_JoV/Videos_of_RestoreNet_Reuslt/out_video_',num2str(small_clip)],['out_video_',num2str(small_clip)]);
    Vedio_clear.open;
    videosa  = eval(['out_video_', num2str(small_clip)]);
    video = reshape(videosa(2,:,:,:,:), [64, 64, 3, 25]);
    for frame = 1:25
        frame_layer_clear = video(1:end, 1:end, 1:end, frame);
        Vedio_clear.writeVideo(frame_layer_clear);
    end
    Vedio_clear.close;
end
