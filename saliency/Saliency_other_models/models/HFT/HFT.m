%%%%%%%%%%%%%%%%%%%%%%%%%%%%  HFT Saliency Computing  %%%%%%%%%%%%%%%%%%%%%
%                                                                         %
% This computes the HFT saliency map for the input image                  %
% The hypercomplex FFT functions are provided by T. Ell[40]               %
%                                                                         %
% Jian Li. March,2011.                                                    %
%                                                                         %
% Email: lijian.nudt@gmail.com; lijian@cim.mcgill.ca                      %
% --You can use and distribute the code freely. If you use the HFT code   %
% --and the corresponding ROC code, please cite our PAMI paper:           %
% --Visual Saliency Based on Scale-Space Analysis in the Frequency Domain %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [SalMap] = HFT(inImg1)
%% Clear history and memory
%clc;clear all;close all;
%% Set path
%rootInfo = what;root = rootInfo.path;root=strcat(root,'\functions\'); addpath(root)
addpath('functions');
%% Initializtion 
param=callHFTParams;
%% Compute saliency for each image
%% Load image
[p1,p2,p3]=size(inImg1);
%% If you have the Ground Truth, you can display them here.
%load the Ground Truth %GndTru=double(imread (strcat(root,'\GroundTruth\','GT1_',num2str(q),'.jpg')));
%close all
%scrsz = get(0,'ScreenSize');
%figure('Position',[scrsz(3)/40 scrsz(4)/3.1 scrsz(3)*19/20 scrsz(4)*6/10])
%subplot(3,8,[1 2 9 10])
%imshow(inImg1/max(inImg1(:)))
%title(['Original Image ',num2str(q)],'fontsize',16,'Color','b')
%annotation('textbox',[.365 .84 0.25 0.15],'String',{'Saliency Map Generation using HFT'},'fontsize',18,'Color','k','FitBoxToText','on');
%subplot(3,8,[7 8 15 16])
%imshow(GndTru/max(GndTru(:)))
%title(['Ground Truth'],'fontsize',16,'Color','b')
%annotation('textbox',[.28 .29 0.25 0.15],'String',{'Choose the Optimal Map from the Spectrum Scale Space'},'fontsize',18,'Color','k','FitBoxToText','on');
%% Resize image to 128*128 %You are encouraged to try other resolutions, or even learn a optimal image scale by training.
inImg1 = imresize(inImg1, [128, 128], 'bilinear');
%% Compute input feature maps
r = inImg1(:,:,1);g = inImg1(:,:,2);b = inImg1(:,:,3);   
I = mean(inImg1,3);I=max(max(r,g),b); % Results by "mean" are slightly different from that by "max"; Please try both of them for better results. We use "mean" for quantitative evaluation.
R = r-(g+b)/2;G = g-(r+b)/2;B = b-(r+g)/2;Y = (r+g)/2-abs(r-g)/2-b;Y(Y<0) = 0;
RG = double(R - G);BY =double(B - Y);
%% Compute the Hypercomplex representation
f = quaternion(0.25*RG, 0.25*BY, 0.5*I);  % the default weigthed for each feature map is 0.25 0.25 0.5
%% Compute the Scale space in frequency domain
[M,N]=size(r);S=MSQF(f,M,N);
%% Find the optimal scale
[H,W,Ch]=size(inImg1);sgm=W*param.SmoothingValue;
for k=1:8; 
      entro(k)=entropy1((S(:,:,k)));     %if run HFT, please use this line;
%     entro(k)=entropy2((S(:,:,k)));     %if run HFT(e), please use this line
end
entro_seq=sort(entro); c=find(entro==entro_seq(1));c=c(1);
SalMap=mat2gray(S(:,:,c));
for k=1:8
    %subplot(3,8,16+k)
    SalMap_k = imfilter(S(:,:,k), fspecial('gaussian',[round(4*sgm) round(4*sgm)],sgm));
    %imshow(SalMap_k,[])
    %if k==c
    %title(['k= ',num2str(k),'  E=',num2str(entro(k))],'fontsize',10,'Color','r')
    %else
    %title(['k= ',num2str(k),'  E=',num2str(entro(k))],'fontsize',10,'Color','b')
    %end
end
%--------------------------------------------------------------------------
%% Postprocessing
%-------------
%incorperate a border cut. A border cut could be employ to alleviate the
%problem caused by the border effect. However the unfairness will be
%introduced when make comparison. In our paper, the border cut is not used.
if param.openBorderCut == 1
SalMap=bordercut(SalMap,param.BorderCutValue);
end
%-------------
sgm=W*param.SmoothingValue;
SalMap = imfilter(SalMap, fspecial('gaussian',[round(4*sgm) round(4*sgm)],sgm));
SalMap = imresize(SalMap, [p1,p2], 'bilinear');
%-------------
%incorperate a global center bias
%However, we think that center bias has little significace, but only inrease
%ROC score. In our paper, the center bias is not incorperated.
if param.setCenterBias == 1
SalMap=CenterBias(SalMap,param.CenterBiasValue);
end
%-------------
SalMap=mat2gray(SalMap);
%subplot(3,8,[4 5 12 13])
%imshow(SalMap,[])
%title('The Saliency Map','fontsize',16,'Color','r')
%imwrite (SalMap,strcat(root,'test\','SalMap_',num2str(q),'.bmp'));
%pause();
%end