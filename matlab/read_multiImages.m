
% Load images
clc;
clear all;
close all;
% for n=1:13
%   images{n} = imread(sprintf('Images/Miscellaneous-USC-DataBase/misc/color_image/%03d.tiff',n));
% end
for K = 1 : 20
  images = imresize(imread((sprintf('%d.jpg', K))),[256, 256]);
  setfigurepos([100 100 900 1000]);
  ax = subplot(4, 5, K);
  imshow(images, 'Parent', ax);

end