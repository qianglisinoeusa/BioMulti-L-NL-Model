
% PATH VISTALAB AND COLORLAB
addpath(genpath('/media/disk/vista/Papers/IMAGING/fMRI_explora/Vistalab'))
addpath(genpath('/media/disk/vista/Papers/2019_Information_Flow_Wilson_Cowan/BioMultilayer_L_NL_convolut/BioMultiLayer_L_NL_color_fast_last/colorlab'))

% PATH INPUT VIDEO AND RESULTS
tray = '/media/disk/databases/BBDD_video_image/statist_pelis_youtube/copyright_free_movies/the_FBI_story_1959/';
tray_blur = '/media/disk/databases/BBDD_video_image/A_Autoencoder_CSFs/video/data_FBI_BIO_blur/';
tray_blur_noise = '/media/disk/databases/BBDD_video_image/A_Autoencoder_CSFs/video/data_FBI_BIO_blur_noise/';

for fot = 1:260 % loop over the seconds
    tic,fot
    % Gather pieces of the selected second
    i_inic = (fot-1)*14*14;
    k = 1;
    sequence = [];
    columna = 1;
    while columna<15
        fila = 1;
        columnita = [];
        while fila<15
              video = read_video_no_plot([tray,'video_FBI_',num2str(i_inic+k),'.avi'], 0 , 25);
              % [fot fila columna]
              k = k+1;
              fila = fila+1;
              columnita = cat(1,columnita,video);
              % size(columnita)
              % pause
        end
        sequence = cat(2,sequence,columnita);
        columna = columna+1;
    end

    % Corrupt every frame of the sequence %%%% DEGRADATION PARAMETERS %%%%
    fs = 30;
    d = 3;      % 6;
    fano = 0.5; % 1.5;
    comp_naif = 0;
    sequence_blur = sequence;
    sequence_blur_noise = sequence;
    for frame=1:25
        im = sequence(:,:,:,frame);
        [im_rgb_filt,im_rgb_filt_noise,im_rgb_naif,im_lms,im_lms_f,im_lms_fn,im_lms_naif] = rgb_retinal_response(im,fs,d,fano,comp_naif);
        sequence_blur(:,:,:,frame) = im_rgb_filt;
        sequence_blur_noise(:,:,:,frame) = im_rgb_filt_noise;
    end

    % Break the big sequence and save the video patches (blur)
    k=1;
    I1 = im2colcube(sequence_blur(:,:,1,:),[32 32],1);
    I2 = im2colcube(sequence_blur(:,:,2,:),[32 32],1);
    I3 = im2colcube(sequence_blur(:,:,3,:),[32 32],1);
    for ii=1:length(I1(1,:))
        II1 = I1(:,ii);
        II2 = I2(:,ii);
        II3 = I3(:,ii);
        II1 = col2imcube(II1,[32 25],[32 32]);
        II2 = col2imcube(II2,[32 25],[32 32]);
        II3 = col2imcube(II3,[32 25],[32 32]);
        vid(:,:,1,:)=II1;
        vid(:,:,2,:)=II2;
        vid(:,:,3,:)=II3;
        name = ['video_FBI_',num2str(i_inic+k)];
        build_color_avi_no_plot(double(vid)/255,25,0,tray_blur,name);
        k = k+1;
    end
    
    % Break the big sequence and save the video patches (blur+noise)
    k=1;
    I1 = im2colcube(sequence_blur_noise(:,:,1,:),[32 32],1);
    I2 = im2colcube(sequence_blur_noise(:,:,2,:),[32 32],1);
    I3 = im2colcube(sequence_blur_noise(:,:,3,:),[32 32],1);
    for ii=1:length(I1(1,:))
        II1 = I1(:,ii);
        II2 = I2(:,ii);
        II3 = I3(:,ii);
        II1 = col2imcube(II1,[32 25],[32 32]);
        II2 = col2imcube(II2,[32 25],[32 32]);
        II3 = col2imcube(II3,[32 25],[32 32]);
        vid(:,:,1,:)=II1;
        vid(:,:,2,:)=II2;
        vid(:,:,3,:)=II3;
        name = ['video_FBI_',num2str(i_inic+k)];
        build_color_avi_no_plot(double(vid)/255,25,0,tray_blur_noise,name);
        k = k+1;
    end
    toc
end