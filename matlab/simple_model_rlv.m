function [mtf, LMSV, filter_oppDN, ws]=simple_model_rlv(img, params)


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                    Digtal Human Visual Cortex Pathway                                             %
%                                       Retina-------LGN---------V1                                                 %
%                                         linear + nonlinear model                                                  %
%                                                                                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% visual pathway simulated from retina - LGN - V1

% load lena.mat
% img = Im;

%warning ('off');
%cd ('/home/qiang/QiangLi/Matlab_Utils_Functional/matlab_human-vision-model_utils/Retina-LGN-V1-models-OnGoing');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Parameters;
load SmithPokornyCones;
load displaySPD;
load displayGamma;
% params.ppd=0.0847;
% params.NM=15;
% params.DIS=500;
% params.N = 64;
% params.variance=0.005;
% params.sampPerDeg=120;
% params.fs=64;
% params.columns_x = params.N;
% params.rows_y = params.N;
% params.frames_t = 1;      
% params.fsx = fs;
% params.fsy = fs;
% params.fst = 24;
% params.to_p = 0.3;
% params.order_p = 1;
% params.sigmax_p = 0.03;  
% params.sigmat_p = 0.1;
% params.to_n = 0.3;
% params.order_n = 1;
% params.sigmax_n = 0.21;
% params.sigmat_n = 0.1;
% params.excit_vs_inhib = 1;
% params.tw=1;
% params.ns_not_used=0;
% params.resid=1;
% params.ns = 3;
% params.no = 4;
% params.fact=10;
% params.expo=0.3;
% params.fact_n=0.2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%mModel with Image
img1 = [img(:,:,1)  img(:,:,2)  img(:,:,3)];
%imgRGB1 = dac2rgb(img1, gammaTable);

[p1,p2,p3]=getPlanes(img1);
%imgRGB1=cat(3,p1,p2,p3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Retina Model;

% Step 1:  Constructure Noise

deg_model='wt_gn';
[Imd, PSF] = KeCoDe_degradation(img1, deg_model,params.variance);
[p1_ p2_ p3_]=getPlanes(Imd);

% Step 2: MTF

f=MTF(params.NM, params.ppd, params.DIS);

F1=fftshift2(f);
f2=real(ifft2(F1));
f3=fftshift(f2);

p1r=conv2(p1_,f3,'same');
p2r=conv2(p2_,f3,'same');
p3r=conv2(p3_,f3,'same');

mtf = cat(3, p1r, p2r, p3r);

p1_f=fftshift(fft2(p1r));
p2_f=fftshift(fft2(p2r));
p3_f=fftshift(fft2(p3r));

aa=20*log(abs(p1_f));
b=20*log(abs(p2_f));
c=20*log(abs(p3_f));

% Step 3: Low_Pass Filters Noise

butter_filter=constructbutterfilter(size(p1r,1),[10 50],5);
[xx,yy] = calccpfov(size(p1r,1));

% cycles per field-of-view
radius = sqrt(xx.^2+yy.^2);  
p1rr = imagefilter(p1r, butter_filter);
p2rr = imagefilter(p2r, butter_filter);
p3rr = imagefilter(p3r, butter_filter);

cc=20*log(abs(fftshift(fft2(butter_filter))));
cc=cc/max(cc(:));


% Step 4: Cone response

rgb2lms = cones'* displaySPD;
rgbWhite = [1 1 1];
whitepoint = rgbWhite * rgb2lms';

imgRGB2=[p1rr p2rr p3rr];
img1LMS = changeColorSpace(imgRGB2,rgb2lms);
[p4,p5,p6]=getPlanes(img1LMS);

p4_f=fftshift(fft2(p4));
p5_f=fftshift(fft2(p5));
p6_f=fftshift(fft2(p6));

% Step 5: VonKries adaptation

imageformat = 'lms';
imageformat = [imageformat '   '];
imageformat = imageformat(1:5);

if (imageformat=='xyz10' | imageformat=='lms10')
    xyztype = 10;
else
    xyztype = 2;
end
[L M S]=getPlanes(img1LMS);
if imageformat(1:3)=='lms'
    opp1 = changeColorSpace(img1LMS, cmatrix('lms2opp'));
    oppwhite = changeColorSpace(whitepoint, cmatrix('lms2opp'));
    whitepoint_post = changeColorSpace(oppwhite, cmatrix('opp2xyz', xyztype));
else
    opp1 = changeColorSpace(img, cmatrix('xyz2opp', xyztype));
end

%Apply Von-Kries in LMS
LMSV(:,:,1) = (whitepoint(1)/whitepoint_post(1))*p4;
LMSV(:,:,2) = (whitepoint(2)/whitepoint_post(2))*p5;
LMSV(:,:,3) = (whitepoint(3)/whitepoint_post(3))*p6;

p7_f=fftshift(fft2(LMSV(:,:,1)));
p8_f=fftshift(fft2(LMSV(:,:,2)));
p9_f=fftshift(fft2(LMSV(:,:,3)));

% Achromatic=L+M : adapting liminiance
% Chromatic= L-M
L_adapt=LMSV(:,:,1)+LMSV(:,:,2);
LMSV_updated=[LMSV(:,:,1) LMSV(:,:,2) LMSV(:,:,3)];

% Step 6: Visual Opponent Channel

img1XaYaZa = changeColorSpace(LMSV_updated, cmatrix('lms2xyz', xyztype));
opp2 = changeColorSpace(img1XaYaZa, cmatrix('xyz2opp', xyztype));

opp2 = MSCN(opp2);
% Step 7: Opponent Filter

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%LGN Model;

% [x,y,t,fx,fy,ft] = spatio_temp_freq_domain(params.N, params.N,1,params.fs,params.fs,1);
% 
% xx1 = x(1,1:4:end); 
% yy1 = x(1,1:4:end);
% 
% xo_p = xx1(end)/2;
% yo_p = yy1(end)/2;
% xo_n = xx1(end)/2;
% yo_n = yy1(end)/2;
% 
% [G,G_excit,G_inhib] = sens_lgn3d_space(params.columns_x, params.rows_y, params.frames_t, params.fsx, params.fsy,params.fst, xo_p, yo_p, params.to_p, params.order_p, params.sigmax_p, params.sigmat_p, xo_n, yo_n, params.to_n, params.order_n, params.sigmax_n, params.sigmat_n, params.excit_vs_inhib);
% G = G_excit/sum(G_excit(:)) - G_inhib/sum(G_inhib(:));
% 
% % Compute the filter from the modulus (to avoid phase shifts due to the location of the RF away from the origin)
% F = abs(fftshift(fft2(G)));
% at = fftshift(fft2(dn_img));  % option1: WB(dn_img) option2: Chromatic images (filter_oppDN)
% B = real(ifft2(ifftshift(at.*F)));
% B = MSCN(B);
% Resp_B = fftshift(abs(fft2(B)));

% [h,CSSFO,CSFT,OE]=csfsso(params.fs, params.N,330.74,7.28,0.837,1.809,1,6.664);
% [csfrg,csfyb]=csf_chrom(params.N,params.fs);
% 
% A0 = mean(mean(dn_img)); % We compute the average luminance in advance to preserve it despite the DC removal by the achromatic CSF
% T0 = mean(mean(p11));
% D0 = mean(mean(p12));
% 
% B = A0+real(ifft2(ifftshift(CSSFO/max(CSSFO(:)).*fftshift(fft2(dn_img-A0)))));
% B = MSCN(B);
% imATDf(:,:,2) = real(ifft2(ifftshift(csfrg/max(CSSFO(:)).*fftshift(fft2(p11)))));
% imATDf(:,:,3) = real(ifft2(ifftshift(csfyb/max(CSSFO(:)).*fftshift(fft2(p12)))));

[k1, k2, k3] = separableFilters(params.sampPerDeg, 2);

k1_f=fftshift(fft2(k1));
k2_f=fftshift(fft2(k2));
k3_f=fftshift(fft2(k3));

[w1, w2, w3] = getPlanes(opp2);

wsize = size(w1);

disp('Filtering BW plane of image1 ...');
p10 = separableConv(w1, k1, abs(k1));
disp('Filtering RG plane of image1 ...');
p11 = separableConv(w2, k2, abs(k2));
disp('Filtering BY plane of image1 ...');
p12 = separableConv(w3, k3, abs(k3));

filter_opp2= cat(3, p10, p11, p12);

p10_f=fftshift(fft2(p10));
p11_f=fftshift(fft2(p11));
p12_f=fftshift(fft2(p12));

p10_f11=20*log(abs(p10_f));
p11_f11=20*log(abs(p11_f));
p12_f11=20*log(abs(p12_f));

%Step 8: Nonlinear(Divisive Normalization)

dn_img = MSCN(p10);  % WB channel 
p11 = MSCN(p11); 
p12 = MSCN(p12); 

filter_oppDN= cat(3, dn_img, p11, p12);
%  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%v1 Model;

%load sensors_V1_64x64_3
% Here, We only care spatial infmation
%[rs,resp,p,ind_gran] = apply_V1_matrix_to_blocks_256_image(B,ggs,indices,0.7,1);

%V1_flow_in=cat(3, B, p11, p12);

%V1_flow_in=[V1_flow_in(:,:,1), V1_flow_in(:,:,2), V1_flow_in(:,:,3)];
V1_flow_in = dn_img;
[w,ind] = buildSFpyr(V1_flow_in, params.ns, params.no-1);
% 
% [h,CSSFO,CSFT,OE]=csfsso(params.fs, params.N,330.74,7.28,0.837,1.809,1,6.664);
% csf = ifftshift(CSSFO);
% csf = csf.*(csf>=params.fact) + params.fact*ones(params.N, params.N).*(csf<=params.fact);
% csf = csf(:);
% dc_factor = max(csf(:));
% csf(1) = dc_factor;
% csf =csf/dc_factor;
% csf1d = csf(params.N/2,params.N/2+1:end);
% 
% [filter_w2] = fourier_to_wavelet_soft(csf,params.ns,params.no);
% [W,ind,indices_si] = make_wavelet_kernel_2(params.N,params.ns,params.no,params.tw,params.ns_not_used,params.resid);
% 
% load images_80
% im1 = imresize(double(im1),[params.N params.N]);
% im2 = imresize(double(im2),[params.N params.N]);
% im3 = imresize(double(im3),[params.N params.N]);
% load images_80_free
% im4 = imresize(double(256*ima1),[params.N params.N]);
% im5 = imresize(double(256*ima2),[params.N params.N]);
% im6 = imresize(double(256*ima3),[params.N params.N]);
% im7 = imresize(double(256*ima4),[params.N params.N]);
% im8 = imresize(double(256*ima5),[params.N params.N]);
% im9 = imresize(double(256*ima6),[params.N params.N]);
% 
% wavelets_filt_natural = zeros(length(filter_w2),9);
% 
% for i=1:9
%     eval(['im = im',num2str(i),';']);
%     [p,ind] = buildSFpyr(im,params.ns,params.no-1);
%     wavelets_filt_natural(:,i)=filter_w2.*p;
% end
% w_average = mean(abs(wavelets_filt_natural),2); 
% w_average = average_deviation_wavelet(w_average,ind);
% 
% csf=filter_w2;

% csf filter in the wavelets domain 
%wf = params.csf.*w;
%w_average = 3*params.w_average;
w_average = params.w_average;
% Saturation repsonse in the wavelets domain
[ws,dsdr] = saturation_f(w,params.expo,w_average,params.fact_n*w_average);


end

