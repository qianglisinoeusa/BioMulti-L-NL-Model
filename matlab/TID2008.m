function [OB, SB] = TID2008()

% ======================================================================
%  Retina-LGN-V1 fitting TID2008 Version 2.0
%  Copyright(c) 2020  Qiang Li
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
% 
%=====================================================================

clc;
clear all;
close all;

%startcol

% TID2008 Database
TID = genpath('/home/qiang/QiangLi/Matlab_Utils_Functional/TID2008');
%!wget https://github.com/shnizelsh/halftoning-evaluator/blob/b3ebf982043f43f72addcf8f4cbac5a8a4e771f3/SFF/TID2008.mat
addpath(genpath('/home/qiang/QiangLi/Matlab_Utils_Functional/matlab_human-vision-model_utils/Retina-LGN-V1-models-OnGoing/'));
cd('/home/qiang/QiangLi/Matlab_Utils_Functional/matlab_human-vision-model_utils/Retina-LGN-V1-models-OnGoing/BioMultiLayer_L_NL_color_fast_last/BioMultiLayer_L_NL_color_fast_last/matlabPyrTools/MEX')
compilePyrTools


try
    load ('TID2008.mat');
catch err
   error('No TID database, go to downoad it\n');
end

dmos = 100-11*tid_MOS;

% Compute parameters
tic
params = parameters_bi_model;
toc

save parameters_BI2020  params 

ScoreSingle = zeros(1700,1);
iPoint = 0;
expo = 2.2; % Changing this to 1 would imply (1) gamma correction ^2.2 and Weber-like raised to 1/2.2 
expo = 1; % Changing this to 1 would imply (1) gamma correction ^2.2 and Weber-like raised to 1/2.2 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
hi=waitbar(0, 'Computing');

indices = [];

for iRef = 1:25%2:4
    %READ A REFERENCE IMAGE
    imNameRef = num2str(iRef,'%02d');
    Ir = imread(['/home/qiang/QiangLi/Matlab_Utils_Functional/TID2008/reference_images/I' imNameRef '.BMP']);
    Ir = imresize(Ir, [256, 256]);
    Iro = double(Ir)/255;
    Ir = Iro.^expo;
    %READ A DISTORTED IMAGE
    for iDis = 1:17
        imNameDis = ['_' num2str(iDis,'%02d')];
        for iLevel = 1:4
            
            index = (iRef-1)*68 + (iDis-1)*4 + iLevel;
            indices = [indices;iRef iDis iLevel index]            
            
            Id = imread(['/home/qiang/QiangLi/Matlab_Utils_Functional/TID2008/distorted_images/I' imNameRef imNameDis '_' num2str(iLevel) '.bmp']);
            Id = imresize(Id, [256, 256]);
            Ido=double(Id)/255;
            Id=Ido.^expo;
            iPoint = iPoint+1;
         
            tic
            %[d0(iPoint),dw(iPoint),dwf(iPoint),dwfs(iPoint),dwfsn(iPoint), dwfsnw(iPoint)]=RLV(255*Ir,255*Id, params);
            [d0(iPoint),dw(iPoint),dwf(iPoint),dwfs(iPoint), dwfsnw(iPoint)]=RLV(255*Ir,255*Id, params);

            toc
            
            waitbar(iPoint/1700);%204
        end
    end   
end

close(hi);


indi = find(indices(:,2)~=0);                            

SB = dmos(indices(indi,end))';    % Subjective Score
OB = d0(indi);                 % Objective Score

%SB =dmos(69:272)';
%OB =d0;

metric_1 = corr(SB, OB, 'type', 'pearson');     % Pearson linear correlation coefficient (without mapping)
metric_2 = corr(SB, OB, 'type', 'spearman');    % Spearman rank-order correlation coefficient
metric_3 = corr(SB, OB, 'type', 'kendall');     % Kendall rank-order correlation coefficient
metrics = [metric_1;metric_2;metric_3];
%figure(1),scatter(OB,SB,'b.');xlabel('objective score'), ylabel('subjective score')
figure(1),subplot(151),plot(OB,SB,'m.'),C=corrcoef(OB, SB);title(['\rho_1 = ',num2str(C(1,2))], 'FontSize', 10, 'FontWeight', 'bold'), xlabel('objective score', 'FontSize', 10), ylabel('subjective score', 'FontSize', 10)

OB = dw(indi);    % d0 dw dwf dwfs dwfsn             % Objective Score
%OB=dw;

metric_1 = corr(SB, OB, 'type', 'pearson');     % Pearson linear correlation coefficient (without mapping)
metric_2 = corr(SB, OB, 'type', 'spearman');    % Spearman rank-order correlation coefficient
metric_3 = corr(SB, OB, 'type', 'kendall');     % Kendall rank-order correlation coefficient
metrics = [metric_1;metric_2;metric_3];
%figure(1),scatter(OB,SB,'b.');xlabel('objective score'), ylabel('subjective score')
figure(1),subplot(152),plot(OB,SB,'m.'),C=corrcoef(OB, SB);title(['\rho_2 = ',num2str(C(1,2))],'FontSize', 10, 'FontWeight', 'bold'), xlabel('objective score', 'FontSize', 10), ylabel('subjective score', 'FontSize', 10)

OB = dwf(indi);    % d0 dw dwf dwfs dwfsn             % Objective Score
%OB = dwf;

metric_1 = corr(SB, OB, 'type', 'pearson');     % Pearson linear correlation coefficient (without mapping)
metric_2 = corr(SB, OB, 'type', 'spearman');    % Spearman rank-order correlation coefficient
metric_3 = corr(SB, OB, 'type', 'kendall');     % Kendall rank-order correlation coefficient
metrics = [metric_1;metric_2;metric_3];
%figure(1),scatter(OB,SB,'b.');xlabel('objective score'), ylabel('subjective score')
figure(1),subplot(153),plot(OB,SB,'m.'),C=corrcoef(OB, SB);title(['\rho_3 = ',num2str(C(1,2))], 'FontSize', 10, 'FontWeight', 'bold'), xlabel('objective score', 'FontSize', 10), ylabel('subjective score', 'FontSize',10)

OB = dwfs(indi);    % d0 dw dwf dwfs dwfsn             % Objective Score
%OB = dwfs;

metric_1 = corr(SB, OB, 'type', 'pearson');     % Pearson linear correlation coefficient (without mapping)
metric_2 = corr(SB, OB, 'type', 'spearman');    % Spearman rank-order correlation coefficient
metric_3 = corr(SB, OB, 'type', 'kendall');     % Kendall rank-order correlation coefficient
metrics = [metric_1;metric_2;metric_3];
%figure(1),scatter(OB,SB,'b.');xlabel('objective score'), ylabel('subjective score')
figure(1),subplot(154),plot(OB,SB,'m.'),C=corrcoef(OB, SB);title(['\rho_4 = ',num2str(C(1,2))],'FontSize', 10, 'FontWeight', 'bold'), xlabel('objective score', 'FontSize', 10), ylabel('subjective score', 'FontSize', 10)
% 
% OB = dwfsn(indi);    % d0 dw dwf dwfs dwfsn             % Objective Score
% OB = dwfsn;
% 
% metric_1 = corr(SB, OB, 'type', 'pearson');     % Pearson linear correlation coefficient (without mapping)
% metric_2 = corr(SB, OB, 'type', 'spearman');    % Spearman rank-order correlation coefficient
% metric_3 = corr(SB, OB, 'type', 'kendall');     % Kendall rank-order correlation coefficient
% metrics = [metric_1;metric_2;metric_3];
% %figure(1),scatter(OB,SB,'b.');xlabel('objective score'), ylabel('subjective score')
% figure(1),subplot(155),plot(OB,SB,'g.'),C=corrcoef(OB, SB);title(['\rho_5= ',num2str(C(1,2))]), xlabel('objective score'), ylabel('subjective score')

OB = dwfsnw(indi);    % d0 dw dwf dwfs dwfsn             % Objective Score
%OB = dwfsnw;

metric_1 = corr(SB, OB, 'type', 'pearson');     % Pearson linear correlation coefficient (without mapping)
metric_2 = corr(SB, OB, 'type', 'spearman');    % Spearman rank-order correlation coefficient
metric_3 = corr(SB, OB, 'type', 'kendall');     % Kendall rank-order correlation coefficient
metrics = [metric_1;metric_2;metric_3];
%figure(1),scatter(OB,SB,'b.');xlabel('objective score'), ylabel('subjective score')
figure(1),subplot(155),plot(OB,SB,'m.'),C=corrcoef(OB, SB);title(['\rho_5= ',num2str(C(1,2))], 'FontSize', 10, 'FontWeight', 'bold'), xlabel('objective score', 'FontSize', 10), ylabel('subjective score',  'FontSize', 10)

% set(gcf,'Units','inches');
% screenposition = get(gcf,'Position');
% set(gcf,...
%     'PaperPosition',[0 0 screenposition(3:4)],...
%     'PaperSize',[screenposition(3:4)]);
% print -dpdf -painters epsFig

%figure(2),plot(OB,SB,'.'),nice_scatter_plot_TID(OB,SB,indices,2)


function [d0,dw,dwf,dwfs,dwfsnw] = RLV(Ir,Id, params)

% ======================================================================
%  RLV Version 2.0
%  Copyright(c) 2020  Qiang Li
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
% INPUT variables:
% Ir:       reference color image
% Id:       distorted color image
% param:    parameters of the model see parameters_bi_model.m
% =====================================================================
% 
% REQUIRED TOOLBOXES
% 
% 
%=======================================================================
% Simple Vision Model

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%DIVIDING EACH IMAGE INTO BLOCKS %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% param.N = 64;
% patches_r = im2col(Ir,[param.N param.N],'distinct');
% patches_d = im2col(Id,[param.N param.N],'distinct');
%@ im2col fucntion only works on gray image not color image.

startPosition = 1;
patchSize = 64;
patchDim = (patchSize^2)*3;
sizeY = size(Ir,1); 
sizeX = size(Ir,2);

gridY = startPosition : patchSize : sizeY-patchSize; 
gridX = startPosition : patchSize : sizeX-patchSize; 
Y = length(gridY); 
X = length(gridX);

Xr = zeros(patchDim, Y*X);
Xd = zeros(patchDim, Y*X);
ij = 0;
for i = gridY
    for j = gridX
        ij = ij+1;
        Xr(:,ij) = reshape( Ir(i:i+patchSize-1, j:j+patchSize-1, 1:3), [patchDim 1] );
        Xd(:,ij) = reshape( Id(i:i+patchSize-1, j:j+patchSize-1, 1:3), [patchDim 1] );
    end
end

Xr = double(Xr);       
Xd = double(Xd);
mXr = mean(Xr);        
mXd = mean(Xd);     
patches_r = Xr-ones(size(Xr,1),1)*mXr;
patches_d = Xd-ones(size(Xd,1),1)*mXd;  

for p = 1:length(patches_r(1,:))
    
    patch_r = reshape(patches_r(:,p),[patchSize, patchSize, 3]);
    patch_d = reshape(patches_d(:,p),[patchSize, patchSize, 3]);
    
    [mtf, LMSV, filter_oppDN,  ws] = simple_model_rlv(patch_r, params);
    [mtfd, LMSVd, filter_oppDNd, wsd] = simple_model_rlv(patch_d, params);
    
    if p==1
        D = zeros(length(mtf(:)),length(patches_r(1,:)));
        DF = zeros(length(mtf(:)),length(patches_r(1,:)));
        DFS = zeros(length(mtf(:)),length(patches_r(1,:)));
        %DFSN = zeros(length(B(:)),length(patches_r(1,:)));
        DFSNW = zeros(length(ws),length(patches_r(1,:)));
        D(:,p) = mtf(:)-mtfd(:);
        DF(:,p) = LMSV(:)-LMSVd(:);
        DFS(:,p) = filter_oppDN(:) - filter_oppDNd(:);
        %DFSN(:,p) = B(:) - Bd(:);
        DFSNW(:,p) = ws-wsd;

    else
        D(:,p) = mtf(:)-mtfd(:);
        DF(:,p) = LMSV(:)-LMSVd(:);
        DFS(:,p) = filter_oppDN(:) - filter_oppDNd(:);
        %DFSN(:,p) = B(:) - Bd(:);
        DFSNW(:,p) = ws-wsd;
    end
    
    sum_exponent = 2;
    
    d0 =  sqrt(sum(abs(Ir(:) - Id(:)).^2));
    dw =  sum(    abs(D(:)).^sum_exponent    ).^(1/sum_exponent);
    dwf =  sum(abs(DF(:)).^sum_exponent).^(1/sum_exponent);
    dwfs =  sum(abs(DFS(:)).^sum_exponent).^(1/sum_exponent);
    %dwfsn =  sum(abs(DFSN(:)).^sum_exponent).^(1/sum_exponent);
    dwfsnw =  sum(abs(DFSNW(:)).^sum_exponent).^(1/sum_exponent);
    
end

