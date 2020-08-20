%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                    Mathmatic Human Visual Cortex Pathway                                          %
%                                        retina-------LGN---------V1                                                %
%                                            linear + nonlinear model                                               %
%                                                                                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                 Dependent hdrvdp and matlabPyrTools_1.4_fixed, ICAPCA,
%                                 BioMultilayer_L_NL_color, Vistalab, Colorlab et al,. Toolbox 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc;
warning ('off');
% cd to the MEX subfolder in the matlabPyrTools subfolder and compile the wavelets library. 
%compilePyrTools
% =================================
% Precompute
%
% precompute common variables and structures or take them from cache
% =================================
%set(0,'DefaultFigureColormap',feval('gray'));


writeline('------------------------------------')
writeline('Prepare parameters')
writeline('Setting the parameters')
writeline('Dependent BioMultilayer_L_NL_color toolbox')


Y_W = 200;
[T_l,Yw,Mbx]=loadsysm(['ciexyz']);
[tm,a,g]=loadmonm('std_crt',Mbx);
Mxyz2lms = xyz2con(eye(3),5)';
W_lms = Mxyz2lms*[Y_W Y_W Y_W]';  % For us the canonnical white is the CIE XYZ (by choosing this as the illuminant white as well, we introduce no VonKries change in the colors)
YWcanon = Y_W;
Tnorm = [260 23 35];


writeline('------------------------------------')
writeline('------------------------------------')
writeline('Precompute')
writeline('precompute common variables and structures or take them from cache')
writeline('------------------------------------')

%=====================================
% Stimulus preparation
% The datasets from calibrated The Barcelona Calibrated Images Database.
%=====================================

writeline('------------------------------------')
writeline('Stimulus preparation')
writeline('Downloading hyperspectral Datasets')
writeline('Loading hyperspectral Datasets')
writeline('------------------------------------')
%
% img - image data (can be multi-spectral)
img=imread('Images/0qADtP.jpg');
%============================================
%Case 2: with hyperspectral image datasets
%load Images/samples_time_40x40_9;
% load im_radiance;

% im_radiance_N = zeros(N,N,length(lambdas));
% for i=1:length(lambdas)
%     img(:,:,i) = 350*imresize(squeeze(im_radiance(:,:,i)),[N N]);  % Check lambda resampling in spec2tri
% end  
%==============================================


width = size(img,2);
height = size(img,1);
img_sz = [height width]; 
img_channels = size(img,3); 

figure(), subplot(1,2,1), imagesc(img), axis off, title('original'), hold on,
%========================================
% Dependent hdrvdp and matlabPyrTools_1.4_fixed toolbox
%=========================================

writeline('------------------------------------')
writeline('Dependent hdrvdp toolbox')
writeline('Dependent matlabPyrTools_1.4_fixed toolbox')
writeline('------------------------------------')


global hdrvdp_cache;

if( any( isnan( img(:) ) ) )
    warning( 'hdrvdp:BadImageData', '%s image contains NaN pixel values', name );
    img(isnan(img)) = 1e-5;
end


%========================================
% Setting parameters
%=========================================
if( ~exist( 'options', 'var' ) )
    options = {};
end

metric_par = hdrvdp_parse_options( options );


% Compute pixels per degree for the viewing conditions
ppd = hdrvdp_pix_per_deg( 24, [size(0,2) size(0,1)], 1 );
pixels_per_degree=ppd;

% Calibrated
color_encoding='srgb-display'

% The parameters overwrite the options
if( ~isempty( pixels_per_degree ) )
    metric_par.pix_per_deg = pixels_per_degree;
end
if( ~isempty( color_encoding ) )
    metric_par.color_encoding = color_encoding;
end

switch lower( metric_par.color_encoding )
    case 'luminance'
        if( img_channels ~= 1 )
            error( 'Only one channel must be provided for "luminance" color encoding' );
        end
        check_if_values_plausible(img, metric_par );
    case 'luma-display'
        if( img_channels ~= 1 )
            error( 'Only one channel must be provided for "luma-display" color encoding' );
        end
        img = display_model( img, 2.2, 99, 1 );
    case 'srgb-display'
        if( img_channels ~= 3 )
            error( 'Exactly 3 color channels must be provided for "sRGB-display" color encoding' );
        end
        img = display_model_srgb(double(img)/255);        
    case 'rgb-bt.709'
        if( img_channels ~= 3 )
            error( 'Exactly 3 color channels must be provided for "rgb-bt.709" color encoding' );
        end
        check_if_values_plausible( img(:,:,2), metric_par );
    case 'xyz'
        if( img_channels ~= 3 )
            error( 'Exactly 3 color channels must be provided for "XYZ" color encoding' );
        end
        img = xyz2rgb( img );
        check_if_values_plausible( img(:,:,2), metric_par );
    case 'generic'
        if( isempty( metric_par.spectral_emission ) )
            error( '"spectral_emission" option must be specified when using the "generic" color encoding' );
        end
end


if( metric_par.surround_l == -1 )
    % If surround_l is set to -1, use the geometric (log) mean of each color
    % channel of the reference image
    metric_par.surround_l = geomean( reshape( reference, [size(reference,1)*size(reference,2) size(reference,3)] ) );
elseif( length(metric_par.surround_l) == 1 )
    metric_par.surround_l = repmat( metric_par.surround_l, [1 img_channels] );
end

if( img_channels == 3 && ~isfield( metric_par, 'rgb_display' ) )
    metric_par.rgb_display = 'ccfl-lcd';
end

if( ~isempty( metric_par.spectral_emission ) )
    [tmp, IMG_E] = load_spectral_resp( [metric_par.base_dir '/' metric_par.spectral_emission] );
elseif( isfield( metric_par, 'rgb_display' ) )
    [tmp, IMG_E] = load_spectral_resp( sprintf( 'emission_spectra_%s.csv', metric_par.rgb_display ) );
elseif( img_channels == 1 )
    [tmp, IMG_E] = load_spectral_resp( 'd65.csv' );
else
    error( '"spectral_emissiom" option needs to be specified' );
end

if( img_channels == 1 && size(IMG_E,2)>1 )
    % sum-up spectral responses of all channels for luminance-only data
    IMG_E = sum( IMG_E, 2 );
end

if( img_channels ~= size( IMG_E, 2 ) )
    error( 'Spectral response data is either missing or is specified for different number of color channels than the input image' );
end
metric_par.spectral_emission = IMG_E;

subplot(1,2,2), imagesc(uint8(img)), axis off, title('Calibrated');
%=======================================
% Load spectral emission curves
%=======================================

writeline('------------------------------------')
writeline('Loading standard spectral emission curves')
writeline('------------------------------------')


rho2 = create_cycdeg_image( img_sz*2, metric_par.pix_per_deg ); % spatial frequency for each FFT coefficient, for 2x image size

if( metric_par.do_mtf )
    mtf_filter = hdrvdp_mtf( rho2, metric_par ); % only the last param is adjusted
else
    mtf_filter = ones( size(rho2) );
end


% Load spectral sensitivity curves
[lambda, LMSR_S] = load_spectral_resp( 'log_cone_smith_pokorny_1975.csv' );
LMSR_S(LMSR_S==0) = min(LMSR_S(:));
LMSR_S = 10.^LMSR_S;

[lam, ROD_S] = load_spectral_resp( 'cie_scotopic_lum.txt' );
LMSR_S(:,4) = ROD_S;

IMG_E = metric_par.spectral_emission;

figure(),   subplot(1, 2, 1), mesh(rho2(1,:),rho2(:,1), rho2), axis off, title('rho2 response'), colorbar
            subplot(1, 2, 2), mesh(mtf_filter), axis off, title('MTF transform'), colorbar

figure(),   subplot(1,2,1), plot(lambda, LMSR_S), title('Log cone smith pokorny 1975')
            subplot(1,2,2), plot(lam, ROD_S), title('Cie scotopic luminance')

% =================================
% Precompute photoreceptor non-linearity/edge detection
% =================================

writeline('------------------------------------')
writeline('Precompute photoreceptor non-linearity')
writeline('------------------------------------')


pn = cache_get( 'pn', [metric_par.rod_sensitivity metric_par.csf_sa], @() create_pn_jnd( metric_par ) );

pn.jnd{1} = pn.jnd{1} * metric_par.sensitivity_correction;
pn.jnd{2} = pn.jnd{2} * metric_par.sensitivity_correction;

figure(),   subplot(2,2,1), plot(pn.Y{1}, pn.jnd{1}, 'y','LineWidth',3),ylabel('response'), xlabel('log space'), title('Cones sensitivity'),
            subplot(2,2,2), plot(pn.Y{2}, pn.jnd{2}, 'r','LineWidth',3),xlabel('log space'),title('Rod sensitivity'),
            subplot(2,2,3), imagesc(pn.jnd{1}), colormap gray, axis off, 
            subplot(2,2,4), imagesc(pn.jnd{2}), colormap gray, axis off;  
figure(),   plot(pn.Y{1}, pn.jnd{1}, 'y','LineWidth',3);
            hold on
            plot(pn.Y{2}, pn.jnd{2}, 'r','LineWidth',3),
            ylabel('response'), xlabel('log space'),
            legend('Cones', 'Rods', 'Location','northwest')
            title('Cones and Rods non-linearlity sensitivity function')

figure(),   subplot(2,3,1), plot(pn.Y{1}, pn.jnd{1}, 'y','LineWidth',3);
            hold on
            plot(pn.Y{2}, pn.jnd{2}, 'r','LineWidth',3),
            legend('Cones', 'Rods', 'Location','northwest'),
            ylabel('response'), xlabel('log space'),title('Cones and Rods Saturation Properties')
            subplot(2,3,2), plot(pn.Y{1}, pn.jnd{1}, 'y','LineWidth',3), title('Cones sensitivity'),
            subplot(2,3,3), plot(pn.Y{2}, pn.jnd{2}, 'r','LineWidth',3), title('Rods sensitivity'),
            subplot(2,3,5), imagesc(pn.jnd{1}), colormap gray, axis off, 
            subplot(2,3,6), imagesc(pn.jnd{2}), colormap gray, axis off;

%===============================================
% Test the cones and rods function
% the non-linearlity of cones and rods
%===============================================

writeline('-----------------------------------------')
writeline('The non-linearlity of cones and rods')
writeline('-----------------------------------------')
Lmax = 150;
gamma = 2;

% Spatial Domain (1 degree sampled at 80 samples/degree -> 80*80 discrete image)
% ------------------------------------------------------------------------------

num_rows = 80;
num_cols = 80;
num_frames =1;
fsx=80;
fsy=80;
fst=0;
[x,y,t,fx,fy,ft] = spatio_temp_freq_domain(num_rows,num_cols,num_frames,fsx,fsy,fst);

%
% SINUSOIDS
L0 = [20, 50, 80, 110, 140];
L1=70;
C=0.8;

% Frequency
f0 = 4;
orient = 10*pi/180;
fx0 = f0*cos(orient);
fy0 = f0*sin(orient);

S = zeros(num_rows,num_cols,length(L0));

% Energy and contrast measures
L=[];
CM_sinus = [];
CRMS_sinus = [];
E_sinus = [];

figure(),colormap gray,
for i = 1:length(L0)
    S(:,:,i) = L0(i) + C*L0(i)*sin(2*pi*(fx0*x+fy0*y));
    subplot(1,length(L0),i),imagesc(x(1,:),y(:,1),S(:,:,i).^(1/gamma),[0 Lmax^(1/gamma)])
    axis square
    if i==1
       xlabel('x (deg)'),ylabel('y (deg)') 
    end
    s = S(:,:,i);
    Cm = (max(s(:))-min(s(:)))/(max(s(:))+min(s(:)));
    Cs = sqrt(2)*std(s(:))/mean(s(:));
    E = mean(s(:).^2);
    title(['C_{RMS}=',num2str(round(100*Cs)/100),  'L=', num2str(L0(i))])
    
    CM_sinus = [CM_sinus Cm];
    CRMS_sinus = [CRMS_sinus Cs];
    E_sinus = [E_sinus E];
    
end
set(gcf,'color',([L1 L1 L1]/Lmax).^(1/gamma))


writeline("===================================================================================")
writeline("=========================Finished Precompute parameters============================")
writeline("===================================================================================")
writeline("===========================retina-------LGN---------V1=============================")


% =================================
% Optical Transfer Function
% =================================
% MTF reduces luminance values

writeline('------------------------------------')
writeline('First Layer --Linear')
writeline('Optical Transfer Function')
writeline('MTF reduces luminance values')  
writeline('------------------------------------')

L_O = zeros( size(img) );
if( metric_par.do_mtf )
        % Use per-channel or one per all channels surround value
        pad_value_1 = metric_par.surround_l( 1 );
        pad_value_2 = metric_par.surround_l( 2 );
        pad_value_3 = metric_par.surround_l( 3 );
        L_O(:,:,1) =  clamp( fast_conv_fft( double(img(:,:,1)), mtf_filter, pad_value_1 ), 1e-5, 1e10 );
        L_O(:,:,2) =  clamp( fast_conv_fft( double(img(:,:,2)), mtf_filter, pad_value_2 ), 1e-5, 1e10 );
        L_O(:,:,3) =  clamp( fast_conv_fft( double(img(:,:,3)), mtf_filter, pad_value_3 ), 1e-5, 1e10 );          
else
        % NO mtf
        L_O(:,:,k) =  img(:,:,k);
end   
L = cat(3,L_O(:,:,1),L_O(:,:,2),L_O(:,:,3));

[L1,C1] = lumin_contrast(L);
[L2,C2] = lumin_contrast(img);
figure(); subplot(131), mesh(20*log(abs(fftshift2(mtf_filter)))), axis off, colorbar, title('MTF filter')
          subplot(132), imagesc(uint8(L)), axis off;    title(['MTF-'  'Lum:',  num2str(L1)])
          subplot(133), imagesc(uint8(img)), axis off;  title(['NMTF-' 'Lum:',  num2str(L2)])  



% =================================
% Decomposition chromatic channel 
% =================================
% Extract red, green, blue channel

writeline('------------------------------------')
writeline('Decomposition chromatic channel')
writeline('Red, Green, Blue channels')  
writeline('------------------------------------')

redChannel = L(:,:,1);
greenChannel = L(:,:,2);
blueChannel = L(:,:,3); 
allBlack = zeros(size(L, 1), size(L, 2), 'uint8');
% Create color versions of the individual color channels.
just_red = cat(3, redChannel, allBlack, allBlack);
just_green = cat(3, allBlack, greenChannel, allBlack);
just_blue = cat(3, allBlack, allBlack, blueChannel);

% Recombine the individual color channels to create the original RGB image again.
% recombinedim1RGB = cat(3, redChannel, greenChannel, blueChannel);
writeline('===================================')
writeline('First Layer --Linear--DCT MODEL')
writeline('Dependent Colorlab && V1_model_DCT_DN_color Toolbox')
writeline('===================================')


xyuv = my_rgb2yuv(L);

YChannel = xyuv(:,:,1);
UChannel = xyuv(:,:,2);
VChannel = xyuv(:,:,3); 

just_Y = cat(3, YChannel, allBlack, allBlack);
just_U = cat(3, allBlack, UChannel, allBlack);
just_V = cat(3, allBlack, allBlack, VChannel);

% convert color space
just_UU=my_yuv2rgb(just_U);
just_VV=my_yuv2rgb(just_V);

% Create color versions of the individual color channels.

fontSize=8;

figure(),
subplot(2, 4, 1), imshow(uint8(L)), title('MTF response', 'FontSize', fontSize),
subplot(2, 4, 2), imshow(just_red), title('red channel', 'FontSize', fontSize),
subplot(2, 4, 3), imshow(just_green), title('green channel', 'FontSize', fontSize),
subplot(2, 4, 4), imshow(just_blue); title('blue channel', 'FontSize', fontSize),
subplot(2, 4, 5),imagesc(xyuv), axis off, title('YUV space', 'FontSize', fontSize),
subplot(2, 4, 6),imagesc(YChannel), axis off, title('Brightness', 'FontSize', fontSize),
subplot(2, 4, 7),imagesc(just_UU), axis off, title('U channel', 'FontSize', fontSize ),
subplot(2, 4, 8),imagesc(just_VV), axis off, title('V channel', 'FontSize', fontSize)
set(gcf, 'Units', 'Normalized', 'OuterPosition', [3, 3, 5, 5]);
% =================================
% Chromatic Transform RBG-XYZ-LMS MODEL
% Photoreceptor spectral sensitivity
% =================================

writeline('------------------------------------')
writeline('Photoreceptor spectral sensitivity')
writeline('------------------------------------')

M_img_lmsr = zeros( img_channels, 4 ); % Color space transformation matrix
for k=1:4
    for l=1:img_channels
        M_img_lmsr(l,k) = trapz( lambda, LMSR_S(:,k).*IMG_E(:,l) );                
    end
end

% Color space conversion for visualization
R_LMSR = reshape( reshape( L_O, width*height, img_channels )*M_img_lmsr, height, width, 4 );

[L_lum, CL]=lumin_contrast(R_LMSR(:,:,1));
[M_lum, CM]=lumin_contrast(R_LMSR(:,:,2));
[S_lum, CS]=lumin_contrast(R_LMSR(:,:,3));
[R_lum, CR]=lumin_contrast(R_LMSR(:,:,4));


figure(),
    subplot(1,5,1),imagesc(R_LMSR(:,:,1)), axis off, title(['L Response-' 'lum:', num2str(L_lum)]) 
    subplot(1,5,2),imagesc(R_LMSR(:,:,2)), axis off, title(['M Response-' 'lum:', num2str(M_lum)])
    subplot(1,5,3),imagesc(R_LMSR(:,:,3)), axis off, title(['S Response-' 'lum:', num2str(S_lum)]) 
    subplot(1,5,4),imagesc(R_LMSR(:,:,1:3)), axis off, title('Cone Response') 
    subplot(1,5,5),imagesc(R_LMSR(:,:,4)), axis off, title(['Rod Response-' 'lum:', num2str(R_lum)]) 

surround_LMSR = metric_par.surround_l * M_img_lmsr;

% =================================
% Adapting luminance
% =================================
writeline('------------------------------------')
writeline('Adapting luminance')
writeline('------------------------------------')


L_adapt = R_LMSR(:,:,1) + R_LMSR(:,:,2);

% =================================
% Photoreceptor non-linearity
% =================================


writeline('------------------------------------')
writeline('Photoreceptor non-linearity')
writeline('------------------------------------')

La = mean( L_adapt(:) );

figure(),   subplot(1,2,1), imagesc(L_adapt), colormap gray, axis off, title('Adapting luminance')
            subplot(1,2,2), imagesc(La),  colormap gray, axis off, title('Mean adapting luminance')



P_LMR = zeros(height, width, 4);
surround_P_LMR = zeros(1,4);
for k=[1:2 4] % ignore S - does not influence luminance   
    if( k==4 )
        ph_type = 2; % rod
        ii = 3;
    else
        ph_type = 1; % cone
        ii = k;
    end
    
    P_LMR(:,:,ii) = pointOp( log10( clamp(R_LMSR(:,:,k), 10^pn.Y{ph_type}(1), 10^pn.Y{ph_type}(end)) ), ...
        pn.jnd{ph_type}, pn.Y{ph_type}(1), pn.Y{ph_type}(2)-pn.Y{ph_type}(1), 0 );
    %figure(22), imagesc(P_LMR(:,:,ii)), colormap gray,axis off, %title('Center photoreceptor non-linearity')    
    surround_P_LMR(ii) = interp1( pn.Y{ph_type}, pn.jnd{ph_type}, ...
        log10( clamp(surround_LMSR(k), 10^pn.Y{ph_type}(1), 10^pn.Y{ph_type}(end)) ) );
    %figure(23), imagesc(surround_P_LMR(ii)),  colormap gray, axis off, %title('Surround photoreceptor non-linearity')
end

% =================================
% Remove the DC component, from 
% cone and rod pathways separately
% =================================

writeline('------------------------------------')
writeline(' Remove the DC component from cone and rod pathways separately')
writeline('------------------------------------')


% TODO - check if there is a better way to do it
% cones
P_C = P_LMR(:,:,1)+P_LMR(:,:,2);
mm = mean(mean( P_C ));
P_C = P_C - mm;
%figure(24), imagesc(P_C),axis off, %title('Remove the DC component. Cones')

% rods
mm = mean(mean( P_LMR(:,:,3) ));
P_R = P_LMR(:,:,3) - mm;
%figure(25), imagesc(P_R), axis off, %title('Remove the DC component. Rods')

figure(),   subplot(2,2,1), imagesc(P_LMR(:,:,ii)), colormap gray,axis off, title('Center photoreceptor response')  
            subplot(2,2,2), imagesc(surround_P_LMR(ii)),  colormap gray, axis off, title('Surround photoreceptor response')
            subplot(2,2,3), imagesc(P_C),axis off, title('Remove the DC component/Cones')
            subplot(2,2,4), imagesc(P_R),axis off, title('Remove the DC component/Rods')              

% =================================
% Achromatic response
% =================================


writeline('------------------------------------')
writeline('Achromatic response')
writeline('------------------------------------')



P = P_C + P_R;
%figure(26), imagesc(P), axis off, title('Achromatic response')

surround_P = surround_P_LMR(1)+surround_P_LMR(2)+surround_P_LMR(3);
%figure(27), imagesc(surround_P), axis off, title('Surround achromatic response')

figure(), subplot(1,2,1), imagesc(P), axis off, title('Achromatic response')
          subplot(1,2,2), imagesc(surround_P), axis off, title('Surround achromatic response')
 
% =================================
% Soc model for contrast
% =================================

writeline('------------------------------------')
writeline('running soc model + DN MODEL')
writeline('dependent knkutils-master toolbox')
writeline('------------------------------------')


im_con = imresize(P,[256, 256]);
[response,cache] = socmodel(im_con);

figure(),
subplot(2,2,1), imagesc(im_con),axis image tight, axis off, colormap(gray); title('Achromatic response'),
subplot(2,2,2), imagesc(reshapesquare(cache.stimulus3)),axis image tight, axis off, colormap(gray), title('Contrast image'),
subplot(2,2,3), imagesc(reshapesquare(cache.stimulus4)),axis image tight; axis off, colormap(gray), title('Second-order contrast'),
subplot(2,2,4), imagesc(reshapesquare(cache.stimulus5)),axis image tight; axis off, colormap(gray), title('Power-law nonlinearity');

% =================================
% Multi-channel decomposition
% nCSF based 
% Wavelets Transform with steerable pyramid 
% =================================

writeline('------------------------------------')
writeline('Wavelets Transform with steerable pyramid')
writeline('Multi-channel decomposition')
writeline('------------------------------------')

P1=reshapesquare(cache.stimulus5);
%[lo0filt,hi0filt,lofilt,bfilts,steermtx,harmonics] = eval(metric_par.steerpyr_filter);
%max_ht = maxPyrHt(size(P), size(lofilt,1));
%[bands.pyr,bands.pind] = buildSpyr( P, min(max_ht,6), metric_par.steerpyr_filter );

[bands.pyr,bands.pind] = buildSpyr(P1, 'auto', metric_par.steerpyr_filter );

%plot the transform result from above
showSpyr(bands.pyr, bands.pind)


bands.sz = ones( spyrHt( bands.pind ) + 2, 1 );
bands.sz(2:end-1) = spyrNumBands( bands.pind );

band_freq = 2.^-(0:(spyrHt( bands.pind )+1)) * metric_par.pix_per_deg / 2;

% CSF-filter the base band
L_mean = mean( L_adapt(:) );
    
BB = pyrBand( bands.pyr, bands.pind, sum(bands.sz) );
% I wish to know why sqrt(2) works better, as it should be 2
rho_bb = create_cycdeg_image( size(BB)*2, band_freq(end)*2*sqrt(2) ); 

csf_bb = hdrvdp_ncsf( rho_bb, L_mean, metric_par );
figure(), surf(csf_bb), title('Neural contrast sensitivity filter')

bb_padvalue = -1; 
bb_padvalue = mean(BB(:));

    
bands.pyr(pyrBandIndices(bands.pind,sum(bands.sz))) = fast_conv_fft( BB, csf_bb, bb_padvalue );


if( 0 )
for b=1:length(bands.sz)
    for o=1:bands.sz(b)
   
        b_ind = sum(bands.sz(1:(b-1)))+o;
        BB = pyrBand( bands.pyr, bands.pind, b_ind );
        rho_bb = create_cycdeg_image( size(BB)*2, band_freq(b)*4 );
        csf_bb = hdrvdp_csf( rho_bb, L_mean, metric_par );
        if( metric_par.surround_l == -1 )
            pad_value = mean(BB(:));
        else
            pad_value = metric_par.surround_l;
        end
    
        bands.pyr(pyrBandIndices(bands.pind,b_ind)) = fast_conv_fft( BB, csf_bb, pad_value );

    end
    
end
end
