% A standard model for foveal detection of spatial contrast 
% Paper: A standard model for foveal detection of spatial contrast 
% AB Watson, AJ Ahumada Journal of Vision 5 (9), 6, 2005


% Model Structure

% Stimuli-Contrast-CSF-Oblique effect-Spatial Aperture-Channel-Minkowski pooling

% Input and output

% input to the model was:  One of the digital stimulus images,
% as provided in the file modelfest-stimuli. 
% Each image has Ny = 256 rows and Nx = 256 columns. 

% The output of the model was: C_ontrast threshold. 


%%%%%%%%%%%%%%%%%Step One%%%%%%%%%%%%%%%%%%%%
%INPUT
% Convert gray_level image to Luminiance
% L0 mean luminiance(30+/-5 cd/m^2)
% % c: contrast
% % g: gain
% L_g=L0(1+c/127(g-128));

% %OUTPUT
% thres=round(log10(c));
% db=20*log10(c);


%%%%%%%%%%%%%%%%%Step Two%%%%%%%%%%%%%%%%%%%%
% Contrast sensitivity filter
% Contrast_Image * radially symmetric CSF(discrete digital finite impulse response (FIR) filter
% created by sampling a one-dimensional CSF in the two-dimensional discrete Fourier transform (DFT) domain)


% Different versions of the CSF
num_rows=64;
num_cols=64;
num_frames=1;
fsx=64;
fsy=64;
fst=1;

fs=40; % Sampling frequency (in cl/deg)
N=256; % Image size (in pixels) 

% Discrete Fourier domain
[F1,F2] = freqspace([N N],'meshgrid');
f=F1(1,:)*fs/2;


[x,y,t,fx,fy,ft] = spatio_temp_DCT_domain(num_rows,num_cols,num_frames,fsx,fsy,fst);



fs=40; % Sampling frequency (in cl/deg)
N=256; % Image size (in pixels) 

% Discrete Fourier domain
[F1,F2] = freqspace([N N],'meshgrid');
f=F1(1,:)*fs/2;

[h,CSSFO,CSFT,OE]=csfsso(fs,N,330.74,7.28,0.837,1.809,1,6.664);

% CHROMATIC CSFs (K. Mullen, Vis. Res. 85) here isotropy is assumed
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[csfrg,csfyb]=csf_chrom(N,fs);



figure(1),
colormap default,
subplot(141),mesh(f,f,(CSSFO)),xlabel('f_x (cpd)'),ylabel('f_y (cpd)'),title('Achromatic CSF'),axis([-35 35 -35 35 0 (200)])
subplot(142),mesh(f,f,(csfrg)),xlabel('f_x (cpd)'),ylabel('f_y (cpd)'),title('Chromatic CSF (RG)'),axis([-35 35 -35 35 0 (200)])
subplot(143),mesh(f,f,(csfyb)),xlabel('f_x (cpd)'),ylabel('f_y (cpd)'),title('Chromatic CSF (YB)'),axis([-35 35 -35 35 0 (200)])
subplot(144),
loglog(f(floor(N/2)+1:end),CSSFO(floor(N/2)+1,floor(N/2)+1:end),'k-',f(floor(N/2)+1:end),csfrg(floor(N/2)+1,floor(N/2)+1:end),'r-',f(floor(N/2)+1:end),csfyb(floor(N/2)+1,floor(N/2)+1:end),'b-');
legend('CSF Achrom','CSF RG','CSF YB'),xlabel('f (cl/deg)'),title('Log Log representation')

figure(2),
subplot(131),colormap gray,subimage(f,f,256*CSSFO/max(max(CSSFO)),repmat(linspace(0,1,256)',1,3)),xlabel('f_x (cl/deg)'),ylabel('f_y (cl/deg)')
subplot(132),colormap gray,subimage(f,f,256*csfrg/max(max(CSSFO)),repmat(linspace(0,1,256)',1,3)),xlabel('f_x (cl/deg)'),ylabel('f_y (cl/deg)')
subplot(133),colormap gray,subimage(f,f,256*csfyb/max(max(CSSFO)),repmat(linspace(0,1,256)',1,3)),xlabel('f_x (cl/deg)'),ylabel('f_y (cl/deg)')


figure(3),
colormap default,
subplot(141),mesh(f,f,(h)),xlabel('f_x (cpd)'),ylabel('f_y (cpd)'),title('Contrast Sensitivity Filter(FIR)'),axis([-35 35 -35 35 0 (200)])
subplot(142),mesh(f,f,(CSFT)),xlabel('f_x (cpd)'),ylabel('f_y (cpd)'),title('DOG'), axis([-35 35 -35 35 0 (200)])
subplot(143),mesh(f,f,(OE)),xlabel('f_x (cpd)'),ylabel('f_y (cpd)'),title('Oblique effect filter'),axis([-35 35 -35 35 0 (200)])


