clear all; clc; close all;

AA = read_video_no_plot('moving_sinusoids/sin_a_st_c_10_fx_1_ft_4.avi', 0, 25);
aa1 = imresize(double(AA)/255, [32 32]); clear AA;

BB = read_video_no_plot('moving_sinusoids/sin_a_st_c_10_fx_1_ft_10.avi', 0, 25);
aa2 = imresize(double(BB)/255, [32 32]); clear BB;

CC = read_video_no_plot('moving_sinusoids/sin_a_st_c_10_fx_2_ft_7.avi', 0, 25);
aa3 = imresize(double(CC)/255, [32 32]); clear CC;

DD = read_video_no_plot('moving_sinusoids/sin_a_st_c_10_fx_3_ft_10.avi', 0, 25);
aa4 = imresize(double(DD)/255, [32 32]); clear DD;

EE = read_video_no_plot('moving_sinusoids/sin_a_st_c_10_fx_6_ft_3.avi', 0, 25);
aa5 = imresize(double(EE)/255, [32 32]); clear EE;

FF = read_video_no_plot('moving_sinusoids/sin_a_st_c_10_fx_13_ft_11.avi', 0, 25);
aa6 = imresize(double(FF)/255, [32 32]); clear FF;

A_a = cat(2, aa1, aa2, aa3, aa4, aa5, aa6);

AA = read_video_no_plot('moving_sinusoids/sin_rg_st_c_10_fx_1_ft_4.avi', 0, 25);
aa1 = imresize(double(AA)/255, [32 32]); clear AA;

BB = read_video_no_plot('moving_sinusoids/sin_rg_st_c_10_fx_1_ft_10.avi', 0, 25);
aa2 = imresize(double(BB)/255, [32 32]); clear BB;

CC = read_video_no_plot('moving_sinusoids/sin_rg_st_c_10_fx_2_ft_7.avi', 0, 25);
aa3 = imresize(double(CC)/255, [32 32]); clear CC;

DD = read_video_no_plot('moving_sinusoids/sin_rg_st_c_10_fx_3_ft_10.avi', 0, 25);
aa4 = imresize(double(DD)/255, [32 32]); clear DD;

EE = read_video_no_plot('moving_sinusoids/sin_rg_st_c_10_fx_6_ft_3.avi', 0, 25);
aa5 = imresize(double(EE)/255, [32 32]); clear EE;

FF = read_video_no_plot('moving_sinusoids/sin_rg_st_c_10_fx_13_ft_11.avi', 0, 25);
aa6 = imresize(double(FF)/255, [32 32]); clear FF;

A_rg = cat(2, aa1, aa2, aa3, aa4, aa5, aa6);

AA = read_video_no_plot('moving_sinusoids/sin_yb_st_c_10_fx_1_ft_4.avi', 0, 25);
aa1 = imresize(double(AA)/255, [32 32]); clear AA;

BB = read_video_no_plot('moving_sinusoids/sin_yb_st_c_10_fx_1_ft_10.avi', 0, 25);
aa2 = imresize(double(BB)/255, [32 32]); clear BB;

CC = read_video_no_plot('moving_sinusoids/sin_yb_st_c_10_fx_2_ft_7.avi', 0, 25);
aa3 = imresize(double(CC)/255, [32 32]); clear CC;

DD = read_video_no_plot('moving_sinusoids/sin_yb_st_c_10_fx_3_ft_10.avi', 0, 25);
aa4 = imresize(double(DD)/255, [32 32]); clear DD;

EE = read_video_no_plot('moving_sinusoids/sin_yb_st_c_10_fx_6_ft_3.avi', 0, 25);
aa5 = imresize(double(EE)/255, [32 32]); clear EE;

FF = read_video_no_plot('moving_sinusoids/sin_yb_st_c_10_fx_13_ft_11.avi', 0, 25);
aa6 = imresize(double(FF)/255, [32 32]); clear FF;

A_yb = cat(2, aa1, aa2, aa3, aa4, aa5, aa6);

%%
A_F = cat(1, A_a, A_rg, A_yb);
implay(A_F)
build_color_avi(A_F,1,5,0,'./','spat_temp_sinus');
%MOV = build_achrom_movie_avi(A_L,0,1,256,1,10,0,'./','spat_temp_sinus');

