function params = parameters_bi_model

% parameters_simple_model sets the parameters of the model.
% Please edit and change!

params.ppd=0.0847;
params.NM=15;
params.DIS=500;  
params.N = 64;
params.variance=0.005;
params.sampPerDeg=64;  %120 bad   80 good    60 very good  40 good
params.fs=64;
params.columns_x = params.N;
params.rows_y = params.N;
params.frames_t = 1;      
params.fsx = 64;
params.fsy = 64;
params.fst = 24;
params.to_p = 0.3;
params.order_p = 1;
params.sigmax_p = 0.03;  
params.sigmat_p = 0.1;
params.to_n = 0.3;
params.order_n = 1;
params.sigmax_n = 0.21;
params.sigmat_n = 0.1;
params.excit_vs_inhib = 1;
params.tw=1;
params.ns_not_used=0;
params.resid=1;
params.ns = 3;
params.no = 4;
params.fact=50;  % 2=0.81527; 5=0.81546; 10=0.81562; 15=0.81567; 20=0.81569; 40=0.8157; 50=0.8157; 60=0.81569
params.expo=0.2;    %0.6=0.799  (0.6 Ok); 0.9=0.79895 (2:4)
params.fact_n=0.4;  %0.2


[h,CSSFO,CSFT,OE]=csfsso(params.fs, params.N,330.74,7.28,0.837,1.809,1,6.664);
csf = ifftshift(CSSFO);
csf = csf.*(csf>=params.fact) + params.fact*ones(params.N, params.N).*(csf<=params.fact);
csf = csf(:);
dc_factor = max(csf(:));
csf(1) = dc_factor;
csf =csf/dc_factor;
csf1d = csf(params.N/2,params.N/2+1:end);

[filter_w2] = fourier_to_wavelet_soft(csf,params.ns,params.no);
[W,ind,indices_si] = make_wavelet_kernel_2(params.N,params.ns,params.no,params.tw,params.ns_not_used,params.resid);

load images_80
im1 = imresize(double(im1),[params.N params.N]);
im2 = imresize(double(im2),[params.N params.N]);
im3 = imresize(double(im3),[params.N params.N]);
load images_80_free
im4 = imresize(double(256*ima1),[params.N params.N]);
im5 = imresize(double(256*ima2),[params.N params.N]);
im6 = imresize(double(256*ima3),[params.N params.N]);
im7 = imresize(double(256*ima4),[params.N params.N]);
im8 = imresize(double(256*ima5),[params.N params.N]);
im9 = imresize(double(256*ima6),[params.N params.N]);

wavelets_filt_natural = zeros(length(filter_w2),9);

for i=1:9
    eval(['im = im',num2str(i),';']);
    [p,ind] = buildSFpyr(im,params.ns,params.no-1);
    wavelets_filt_natural(:,i)=filter_w2.*p;
end
w_average = mean(abs(wavelets_filt_natural),2); 
w_average = average_deviation_wavelet(w_average,ind);

csf=filter_w2;

params.csf=csf;
params.w_average=w_average;



