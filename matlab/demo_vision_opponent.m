clear all
close all
clc


startcol

img_raw=imresize(imread('Images/parrot.png'), [256, 256]);
img = double(img_raw)/255;

% True color to indexed image + palette
[im_index,n]=true2pal(img,1500);

% Palette of digital counts to tristimulus values (gamma calibration)
Txyz=val2tri(n,Yw,tm,a,g);
figure(34), colordgm(Txyz,1,T_l,Yw,'symb','s','sizes(3)',8,'showtriang',{2,tm})
figure(35), colorspc(Txyz,1,T_l,Yw,'showvectors',1,'symb','<','sizes(3)',5,'showtriang',{3,tm})

% CIE XYZ Tristimulus images
imXYZ = pal2true(im_index,Txyz);

figure(1),subplot(131),colormap gray,imagesc(imXYZ(:,:,1)),title('X')
subplot(132),colormap gray,imagesc(imXYZ(:,:,2)),title('Y')
subplot(133),colormap gray,imagesc(imXYZ(:,:,3)),title('X')

% Matrix from XYZ to LMS and XYZ to ATD (Ingling and Tsou) 
LMS=xyz2con([1 0 0;0 1 0;0 0 1],5);
Mxyz2lms = LMS';
ATD=xyz2atd([1 0 0;0 1 0;0 0 1],5);
Mxyz2atd = ATD';
figure,plot(T_l(:,1),T_l(:,2),'r-',T_l(:,1),T_l(:,3),'g-',T_l(:,1),T_l(:,4),'b-')
T_lms = Mxyz2lms*(T_l(:,2:4)');
T_atd = Mxyz2atd*(T_l(:,2:4)');
figure,plot(T_l(:,1),T_lms(1,:),'r-',T_l(:,1),T_lms(2,:),'g-',T_l(:,1),T_lms(3,:),'b-')
figure,plot(T_l(:,1),T_atd(1,:),'r-',T_l(:,1),T_atd(2,:),'g-',T_l(:,1),T_atd(3,:),'b-')

% Transform the colors from XYZ to ATD
Tatd = Mxyz2atd*Txyz';

T_achromatic = [Tatd(1,:);0*Tatd(1,:);0*Tatd(1,:)];
T_T = [mean(Tatd(1,:))*ones(size(Tatd(1,:)));Tatd(2,:);0*Tatd(1,:)];
T_D = [mean(Tatd(1,:))*ones(size(Tatd(1,:)));0*Tatd(2,:);Tatd(3,:)];

imATD = pal2true(im_index,Tatd');

figure,imagesc(img)
figure,subplot(131),colormap gray,imagesc(imATD(:,:,1)),title('A')
subplot(132),colormap gray,imagesc(imATD(:,:,2)),title('T')
subplot(133),colormap gray,imagesc(imATD(:,:,3)),title('D')

% Transforms Achromatic colors and 'fake' chromatic colors to digital counts to be displayed
Txyz_a = inv(Mxyz2atd)*T_achromatic;
Txyz_t = inv(Mxyz2atd)*T_T;
Txyz_d = inv(Mxyz2atd)*T_D;

[nA,saturat,Tn]=tri2val(Txyz_a',Yw,tm,a,g,8);
[nT,saturat,Tn]=tri2val(Txyz_t',Yw,tm,a,g,8);
[nD,saturat,Tn]=tri2val(Txyz_d',Yw,tm,a,g,8);

imA = pal2true(im_index,nA);
imT = pal2true(im_index,nT);
imD = pal2true(im_index,nD);

figure,subplot(131),image(imA),title('A')
subplot(132),image(imT),title('T')
subplot(133),image(imD),title('D')
