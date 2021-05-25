% ======================================================================
%  Simple FVision FModel (SFF) TotalCorrelationEstimate Version 2.0
%  Copyright(c) 2020  Qiang Li & Jesus ;-)
%  All Rights Reserved.
%  qiang.li@uv.es
% ----------------------------------------------------------------------
% Permission to use, copy, or modify this software and its documentation
% for educational and research purposes only and without fee is here
% granted, provided that this copyright notice and the original authors'
% names appear on all copies and supporting documentation. This program
% shall not be used, rewritten, or adapted as the basis of a commercial
% software or hardware product without first obtaining permission of the
% authors. The authors make no representations about the suitability of
% this software for any purpose. It is provided "as is" without express
% or implied warranty.
%----------------------------------------------------------------------
% =====================================================================
% 
% REQUIRED TOOLBOXES
% 
%    * Simple vision model
%    * Matlabpyrtools
%    * Vistalab
%    * RBIG
%    * KL ( Ivan Marin-Franch )
% 
%=====================================================================

clc;
clear all;
close all;

% Vlfeat
%addpath(genpath('/media/disk/users/qiangli/matlab-vision-science-toolbox/vlfeat/'));
% run('/media/disk/users/qiangli/matlab-vision-science-toolbox/vlfeat/toolbox/vl_setup');
% Matconvnet
% cd ('/media/disk/users/qiangli/matlab-vision-science-toolbox/matconvnet-1.0-beta17/matlab')
% setenv('MW_NVCC_PATH','/usr/local/cuda-10.0/bin');
% run vl_compilenn ;
% 
% cd ('/media/disk/users/qiangli/matlab-vision-science-toolbox/')
%% load embedding
load('imagenet_val_embed.mat'); % load x (the embedding 2d locations from tsne)
x = bsxfun(@minus, x, min(x));
x = bsxfun(@rdivide, x, max(x));

%% create an embedding image

S = 1300; % size of full embedding image       ---->3000
G = zeros(S, S, 3, 'uint8');
s = 50; % size of every single image

Ntake = 1300;   % the number of total image

%% load validation image filenames

% cee = makecform('cmyk2srgb'); % Only need to be called once
data.folder= '/home/qiang/QiangLi/Python_Utils_Functional/FixaTons/WECSF/mit1003/chromatic';
data.type = 'jpeg';
files = dir(fullfile(data.folder,sprintf('*.%s',data.type)));
color_data = {};

for i = 1:Ntake%length(files)
   fprintf('Loading image %04d of %04d\n',i,Ntake);   %length(files)
   
   if mod(i, 100)==0
        fprintf('%d/%d...\n', i, Ntake);
   end
    
    % location
    a = ceil(x(i, 1) * (S-s)+1);
    b = ceil(x(i, 2) * (S-s)+1);
    a = a-mod(a-1,s)+1;
    b = b-mod(b-1,s)+1;
    if G(a,b,1) ~= 0
        continue % spot already filled
    end
   
   %vl_imreadjpeg/imread
   img = imread(fullfile(data.folder,files(i).name));
   img = imresize(img, [s, s]);
   %img = repmat(img, [1, 1, 3]);
   [h,w,c] = size(img);
   imsize = size(img);
   if (imsize(1)>1 & prod(imsize(2:length(imsize)))>3)  
     dimension = 2;
   else
     dimension = 1;
   end
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
   G(a:a+s-1, b:b+s-1, :) = img;
   
end

%fs = textread('val_imgs_med.txt', '%s');
%N = length(fs);

%for i=1:Ntake
%    
%    if mod(i, 100)==0
%        fprintf('%d/%d...\n', i, Ntake);
%    end
%    
%    % location
%    a = ceil(x(i, 1) * (S-s)+1);
%    b = ceil(x(i, 2) * (S-s)+1);
%    a = a-mod(a-1,s)+1;
%    b = b-mod(b-1,s)+1;
%    if G(a,b,1) ~= 0
%        continue % spot already filled
%    end
    
%    I = imread(fs{i});
%    if size(I,3)==1, I = cat(3,I,I,I); end
%    I = imresize(I, [s, s]);
    
%    G(a:a+s-1, b:b+s-1, :) = I;
    
%end

imshow(G);

%%
imwrite(G, 'cnn_embed_2k.jpg', 'jpg');

%% average up images
% % (doesnt look very good, failed experiment...)
% 
% S = 1000;
% G = zeros(S, S, 3);
% C = zeros(S, S, 3);
% s = 50;
% 
% Ntake = 5000;
% for i=1:Ntake
%     
%     if mod(i, 100)==0
%         fprintf('%d/%d...\n', i, Ntake);
%     end
%     
%     % location
%     a = ceil(x(i, 1) * (S-s-1)+1);
%     b = ceil(x(i, 2) * (S-s-1)+1);
%     a = a-mod(a-1,s)+1;
%     b = b-mod(b-1,s)+1;
%     
%     I = imread(fs{i});
%     if size(I,3)==1, I = cat(3,I,I,I); end
%     I = imresize(I, [s, s]);
%     
%     G(a:a+s-1, b:b+s-1, :) = G(a:a+s-1, b:b+s-1, :) + double(I);
%     C(a:a+s-1, b:b+s-1, :) = C(a:a+s-1, b:b+s-1, :) + 1;
%     
% end
% 
% G(C>0) = G(C>0) ./ C(C>0);
% G = uint8(G);
% imshow(G);

%% do a guaranteed quade grid layout by taking nearest neighbor

S = 1003; % size of final image
G = zeros(S, S, 3, 'uint8');
s = 50; % size of every image thumbnail
xnum = S/s;
ynum = S/s;
N = Ntake;
used = false(N, 1);

qq=length(1:s:S);
abes = zeros(qq*2,2);
i=1;
for a=1:s:S
    for b=1:s:S
        abes(i,:) = [a,b];
        i=i+1;
    end
end
%abes = abes(randperm(size(abes,1)),:); % randperm
size(abes,1)

for i=1:size(abes,1)
    a = abes(i,1);
    b = abes(i,2);
    %xf = ((a-1)/S - 0.5)/2 + 0.5; % zooming into middle a bit
    %yf = ((b-1)/S - 0.5)/2 + 0.5;
    xf = (a-1)/S;
    yf = (b-1)/S;
    dd = sum(bsxfun(@minus, x, [xf, yf]).^2,2);
    dd(used) = inf; % dont pick these
    [dv,di] = min(dd); % find nearest image

    used(di) = true; % mark as done
    
    %I = imread(fs{di});
    %if size(I,3)==1, I = cat(3,I,I,I); end
    %I = imresize(I, [s, s]);
    
    %vl_imreadjpeg/imread
    img = imread(fullfile(data.folder,files(i).name));
    %And then in each iteration of the loop over images: 
    img = imresize(img, [s, s]);
    [h,w,c] = size(img);
    if c<2
        img = repmat(img, [1, 1, 3]);    
    end
    imsize = size(img);
    if (imsize(1)>1 & prod(imsize(2:length(imsize)))>3)  
        dimension = 2;
    else
        dimension = 1;
    end

    G(a:a+s-1, b:b+s-1, :) = img;

    if mod(i,100)==0
        fprintf('%d/%d\n', i, size(abes,1));
    end
end

imshow(G);

%%
% imwrite(G, 'cnn_embed_full_2k.jpg', 'jpg');
