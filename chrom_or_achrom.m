function SA3 = chrom_or_achrom(path, video)

    addpath(genpath('/home/qiangli/Matlab/colorlab_v3'))
    addpath(genpath('/home/qiangli/Matlab/vistalab'))

    tic
    video = read_video_no_plot([path, video], 0 , 30);
    %video = read_video('/home/qiangli/Matlab/color_q_j/jesus2/color/crom1.mp4', 0 , 30);
    %video = read_video('/home/qiangli/Matlab/color_q_j/jesus2/color/crom2.mp4', 0 , 30);

    video = video(1:3:end,1:3:end,:,:);

    Mrgb2xyz = [0.4162    0.3148    0.2328;0.2311    0.6899    0.0790;0.0190    0.0786    0.9057];

    s = size(video(:,:,1,1));
    video=double(video);
    video_xyz=video;
    colores = zeros(prod(s),3,30);
    mean_color_xyz = [0 0 0];
    for i=1:30
        video_xyz(:,:,:,i) = transf_color_image(video(:,:,:,i),Mrgb2xyz);
        colors = reshape(video_xyz(:,:,:,i),[prod(s) 3]);
        colores(:,:,i) = colors;
        mean_color_xyz = mean_color_xyz + mean(colors);
    end

    mean_color_xyz/sum(mean_color_xyz);

    video_rgb = video_xyz;
    SA3 = [];
    for i=1:30
        colors = colores(:,:,i);
        colores_lab(:,:,i) = xyz2lab(colors,mean_color_xyz);

        Satur3 = mean(  sqrt( squeeze(colores_lab(:,2,i)).^2+squeeze(colores_lab(:,3,i)).^2 ) );
        SA3 = [SA3 mean(Satur3)];   

        colores_xyz_adapt(:,:,i) = lab2xyz(colores_lab(:,:,i),[mean_color_xyz(2) mean_color_xyz(2) mean_color_xyz(2)]);

        frame_xyz = reshape(colores_xyz_adapt(:,:,i),[s(1) s(2) 3]);
        video_rgb(:,:,:,i) = transf_color_image(frame_xyz,inv(Mrgb2xyz));

    end
    SA3 = mean(SA3);
    toc

end

