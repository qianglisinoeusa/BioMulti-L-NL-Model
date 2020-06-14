%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                    Digtal Human Visual Cortex Pathway                                             %
%                                       Retina-------LGN---------V1                                                 %
%                                         linear + nonlinear model                                                  %
%                                                                                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                 Dependent hdrvdp and matlabPyrTools_1.4_fixed, ICAPCA,
%                                 BioMultilayer_L_NL_color, Vistalab, Colorlab et al,. Toolbox 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%UPDATED CODE AND FUSIOIN MORE FUNCTION INTO NEXT CLEAN CODE
% Copyright (c) GNU 3
%Author: QiangLi
%Unit: University of Valencia
%Time: 2020,Spain
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%WARNING: IF YOU WANT TO RUN SPECTRAL_ATD_COLOR SCRIPTS IN LOCAL COMPUTER. AND IF
%YOU USE PARAMETERS TO PAREPARE PRERMETERS, YOU SHOULD SHUT OFF OTHER TASK IN YOUR
%COMPUTER, YOU SHOULD CLEAR, CLOSE ,CLC ALL TASK THEN IT CAN RUN SMOOTHLY.


% THE CODE USED TO SIMULATE RETINA-LGN-CORTEX FUNCTION.  
% UNDERSTAND BRAIN FROM DIGITAL VIEWS 
% TEH MODEL NOT BEST ONE, WE NEED TO OPTIMIZAE IT IN REAL-TIME

clear all;
close all;
clc;
warning ('off');

startcol
cd ('/home/qiang/QiangLi/Matlab_Utils_Functional/matlab_human-vision-model_utils/Retina-LGN-V1-models-OnGoing');

global figurepath
figurepath='Images/Result/';
%=====================================================================
% Aux function for HSV
% Dependent SCIELAB, Colorlab, BioMultilayer_L_NL_color, Vistalab 
%=====================================================================
Number_CPUs=8;
parpool('local', Number_CPUs);

disp('***************************************************************************');
disp('***************************************************************************');
disp('***************************************************************************');
disp('                    ... retina-------LGN---------V1 ...                    ')
disp('***************************************************************************');
disp('***************************************************************************');
disp('***************************************************************************');

cover1=imread('Images/cover.png'); figure, imagesc(cover1); axis off; title('Respect Life')
cover2=imread('Images/retina-LGN-V1.png'); figure, imagesc(cover2); axis off; title('RetinaCortex')

%Now, two type case
datatype='natureimage';
%datatype='spectraldataset';
%datatype='ICAminiCortex';

switch lower(datatype)
    
    case 'natureimage'
        % Load Image
        data.folder= 'Images/Miscellaneous-USC-DataBase/misc/color_image';
        data.type      = 'png';
        % collect data
        files = dir(fullfile(data.folder,sprintf('*.%s',data.type)));
        color_data = {};
        for i = 1:length(files)
            fprintf('Processing image %04d of %04d\n',i,length(files));   
            % read the image
            img = imresize(imread(fullfile(data.folder,files(i).name)), [256, 256]);
            img = im2double(img);
            [h,w,c] = size(img);
            imsize = size(img);
            if (imsize(1)>1 & prod(imsize(2:length(imsize)))>3)   % 2-D images
              dimension = 2;
            else
              dimension = 1;
            end
            % use the whole image, i.e. all pixels
            samples = 1:(h*w);
            timg = reshape(img,[h*w c]);
            color_data{i} = timg(samples,:);
        end
        %More noloss image can be download from Kodak Lossless True Color Image Suite
        %Link addresses: http://r0k.us/graphics/kodak/ or more color/gray image 
        %can be found in the Image folder.
        for i=1:size(color_data,2)
            img_raw=reshape(color_data{1,i}, 256, 256, 3);

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%For loop;
            img=img_raw;
            % True color to indexed image + palette
            [im_index,n]=true2pal(img);

            [counts, binLocations]=imhist(img);
            pdf_=counts/numel(img);
            % Computing derivative of log-pdf
            diff_pdf=diff(log(pdf_));

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            % Nature image statistical
            img1 = [img(:,:,1)  img(:,:,2)  img(:,:,3)];
            im2col_ = im2col(img1,[6 1],'distinct');
            mean_im2col = mean(im2col_')';

            C = zeros(6,6);
            M=length(im2col_(1,:));
            for i=1:M
                C = C + (im2col_(:,i)-mean_im2col)*(im2col_(:,i)-mean_im2col)';
            end
            C = (1/M)*C;
            C3 = (1/M)*(im2col_-repmat(mean_im2col,[1 M]))*(im2col_-repmat(mean_im2col,[1 M]))';
            CC3 = cov(im2col_');

            [B,L]=eig(C);
            values = diag(L);
            [vs,inds]=sort(values,'descend');
            Bs = B;
            for i=1:6
                Bs(:,i) = B(:,inds(i));
            end
            Ls = diag(vs);

            h=figure;
            subplot(141),colormap gray,imagesc(C), title('Covariance');
            subplot(142),colormap gray,imagesc(Bs), title(' Eigenvectors');
            subplot(143),colormap gray,imagesc(abs(Ls).^0.3), title('Eigenvalues');
            subplot(144),colormap gray,imagesc(Bs'), title('Eigenvectors');
            suptitle('Stimuli Properties');
            saveas(h,sprintf('FIG1%d.png',i));
            pause(1)
            close all;

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % PCA
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            pca_ = im2col(img1,[32 32],'sliding');
            mean_a = mean(pca_')';
            M=length(pca_(1,:));
            CC = (1/M)*(pca_-repmat(mean_a,[1 M]))*(pca_-repmat(mean_a,[1 M]))';

            [BB,LL]=eig(CC);
            values = diag(LL);
            [vs,inds]=sort(values,'descend');
            BBs = BB;
            for i=1:32*32
                BBs(:,i) = BB(:,inds(i));
            end
            LLs = diag(vs);

            auto_corr_func = reshape(CC(1024/2-16,:),[32 32]);

            h=figure;
            subplot(241),colormap gray,imagesc(CC); freezeColors
            subplot(242),colormap gray,imagesc(BBs); freezeColors
            subplot(243),colormap gray,imagesc(abs(LLs).^0.3); freezeColors
            subplot(244),colormap gray,imagesc(BBs'); freezeColors
            subplot(245),colormap gray,mesh(CC); freezeColors
            subplot(246),colormap gray,mesh(BBs); freezeColors
            subplot(247),colormap gray,mesh(abs(LLs).^0.3); freezeColors
            subplot(248),colormap gray,mesh(BBs'); freezeColors
            suptitle('Image Properties');
            saveas(h,sprintf('FIG2%d.png',i));
            first_7_functions = disp_patches(BBs(:,1:7),7)

            first_144_functions = disp_patches(BBs(:,1:144),12);
            h=figure;
            subplot(141),colormap gray,imagesc(CC), freezeColors
            subplot(142),colormap gray,mesh(auto_corr_func), freezeColors
            subplot(143),colormap gray,imagesc(first_144_functions), freezeColors
            subplot(144),colormap gray,semilogy(vs), freezeColors
            suptitle('Neighbor sensors wiring densely for efficient coding!!!');
            saveas(h,sprintf('FIG3%d.png',i));

            h=figure;
            subplot(121),colormap gray,imagesc(first_144_functions);title('No Remove DC')
            subplot(122),colormap gray,imagesc(removeDC(first_144_functions)); title('Remove DC');
            saveas(h,sprintf('FIG4%d.png',i));
            pause(1)
            close all;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % POWER SPECTRUM ANALYSIS 
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % The correlation of adjancy pixel in the nature image statistics 
            % Joint  distributions  of  image  pixel  intensities  separated  
            % by  three  different distances.
            r_xy=AdjancyCorrPixel(img);
            r_x2y=Adjancy2CorrPixel(img);
            r_x4y=Adjancy4CorrPixel(img);
            % Autocorrelation function between nearby pixels
            TR_xy_updated=TAdjancyCorrPixel(img);

            h=figure;
            plot([TR_xy_updated{:}]', 'b-');
            xlabel('Spatial separation(pixels)');
            ylabel('Correlation');
            ylim([0,1]);
            xlim([0,40]);
            saveas(h,sprintf('FIG5%d.png',i));

            %compute 2D Fourier transform of the image
            imf=fft2(img);
            %take phase
            imf_phase=angle(img)
            %take amplitude
            imf_abs=abs(img)

            %size of image
            fouriersize=size(img,2);

            %Partly code adapted from code by Bruno Olshausen
            imf=fftshift(imf);
            im_pf=abs(imf);
            im_pf=im_pf(:,:,1);
            f=-fouriersize/2:fouriersize/2-1;

            %plot log-log 1-d crosssection of power spectrum
            Pf=rotavg(im_pf);
            freq=[0:fouriersize/2]';
            logPf=log10(Pf(2:fouriersize/2));
            logfreq=log10(freq(2:fouriersize/2));


            %plot 2d log of power spectrum
            h=figure;
            subplot(121); mesh(20*log10(im_pf)); axis off; title('log power spectrum');
            subplot(122); plot(logfreq,logPf, 'r-'); xlabel('logfreq'); ylabel('logPf'); title('power spectrum(log)')
            suptitle('Computing power spectra')
            saveas(h,sprintf('FIG6%d.png',i));
            pause(1)
            close all;

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            % Define sample per degree value/calculate dependent on view distance and monitor!
            sampPerDeg=120;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %  sRGB→CIEXYZ→opponent representation.
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %  Phase I: Gamma Correction Before Retina
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            load displayGamma
            imgRGB1 = dac2rgb(img1,gammaTable);
            [counts, binLocations]=imhist(imgRGB1);
            pdfA=counts/numel(imgRGB1);
            diff_pdf_=diff(log(pdfA));

            h=figure; 

            subplot(221); plot(binLocations, log(pdf_), 'r-', 'LineWidth', 2); 
            xlabel('Bin'); ylabel('Probability Density');
            title('Marginal densities(log domain)');
            subplot(222); plot(binLocations(2:256,1), diff_pdf, 'r-', 'LineWidth', 2); 
            xlabel('Bin'); ylabel('Derivative Probability Density');
            title('Derivative of Marginal densities(log domain)');
            subplot(223); plot(binLocations, log10(pdfA), 'b-', 'LineWidth', 2); 
            xlabel('Bin'); ylabel('Probability Density');
            title('Marginal densities(log domain) after calibr');
            subplot(224); plot(binLocations(2:256,1), diff_pdf_, 'b-', 'LineWidth', 2); 
            xlabel('Bin'); ylabel('Probability Density');
            title('Derivative of Marginal densities(log domain) after calibr');
            saveas(h,sprintf('FIG7%d.png',i));
            
            [p1,p2,p3]=getPlanes(imgRGB1);
            imgRGB1_updated=cat(3,p1,p2,p3);

            h=figure;

            subplot(331), imagesc(img);
            axis square;
            axis off;
            colormap('default');
            title('Input');
            freezeColors

            subplot(334), imagesc(img(:,:,1));
            axis square;
            axis off;
            colormap( gray(128)*diag([1,0,0]));
            title({'\color{red}R channel'});
            freezeColors 

            subplot(335), imagesc(img(:,:,2));
            axis square;
            axis off;
            colormap( gray(128)*diag([0,1,0]));
            title({'\color{green}G channel'});
            freezeColors 

            subplot(336), imagesc(img(:,:,3));
            axis square;
            axis off;
            colormap( gray(128)*diag([0,0,1]));
            title({'\color{blue}B channel'});
            freezeColors 

            subplot(337), imagesc(p1);
            axis square;
            axis off;
            title({'\color{red}Calibr R channel'});
            colormap( gray(128)*diag([1,0,0]));
            freezeColors 

            subplot(338), imagesc(p2);
            axis square;
            axis off;
            colormap( gray(128)*diag([0,1,0]));
            title({'\color{green}Calibr G channel'});
            freezeColors 

            subplot(339), imagesc(p3);
            axis square;
            axis off;
            colormap( gray(128)*diag([0,0,1]));
            title({'\color{blue}Calibr B channel'});
            freezeColors 

            suptitle('Phase I Before Retina, Gamma Correction')
            set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 1, 0.96]);
            saveas(h,sprintf('FIG8%d.png',i));
            pause(1)
            close all;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            principle=imread('Images/what does the retina know about natural scene.png');
            figure; imshow(principle); axis off; title('Retina Model');
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Constructure Noise - No Blur+Gaussian White Noise
            % Dependent KeCoDe Toolbox See External
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            deg_model='wt_gn';
            variance=0.005;

            [Imd,PSF] = KeCoDe_degradation(imgRGB1, deg_model,variance);

            h=figure;
            colormap gray,imshow(Imd);axis square,
            title(['Gaussian White Noise  \sigma^2=',num2str(variance)])
            rmse=sqrt(sum(sum((img1-Imd).^2)))/256;
            [rmse,SSIM,d_l,d_DCT] = computing_distortions(img1,Imd);
            xlabel({['MSE^{1/2} = ',num2str(round(rmse*10)/10),'  SSIM = ',num2str(round(SSIM*100)/100)],['d_{DCT} = ',num2str(round(d_DCT*10)/10)]});
            saveas(h,sprintf('FIG9%d.png',i));
            
            h=figure;
            subplot(121),colormap gray,imshow(imgRGB1);axis square,
            title('Calibr');
            subplot(122),colormap gray,imshow(Imd);axis square,
            title(['Gaussian White Noise  \sigma^2=',num2str(variance)])
            xlabel({['MSE^{1/2} = ',num2str(round(rmse*10)/10),'  SSIM = ',num2str(round(SSIM*100)/100)],['d_{DCT} = ',num2str(round(d_DCT*10)/10)]});
            saveas(h,sprintf('FIG10%d.png',i));
            pause(1)
            close all;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %      Phase II:  Modulation Transfer Function                          %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % 500 --View Distance
            % 15 -- size
            % 0.0847 -- Delta for frequency
            % This function returns a low pass filter that represents the eye. IN
            % this case the viewing distance has been set to 500 mm and the points
            % size to 0.0847/0.0411=ppd. Note that the size of the point corresponds to a pressure
            % in 300 dpi. (0.0847 = 1/300 * 25.4 mm)
            [p1_ p2_ p3_]=getPlanes(Imd);

            h=figure;
            subplot(131), imagesc(p1_),
            colormap( gray(128)*diag([1,0,0])), axis off, axis square,
            title({'\color{red}R Channel+noise'});
            freezeColors

            subplot(132), imagesc(p2_),
            colormap( gray(128)*diag([0,1,0])), axis off, axis square,
            title({'\color{green}G Channel+noise'});
            freezeColors
                    
            subplot(133), imagesc(p3_),
            colormap( gray(128)*diag([0,0,1])), axis off, axis square,
            title({'\color{blue}B Channel+noise'});
            freezeColors
            saveas(h,sprintf('FIG11%d.png',i));


            %f=MFT(15,0.0847,500);
            ppd=0.0847;
            N=15;
            DIS=500;

            f=MTF(N,ppd,DIS);

            F1=fftshift2(f);
            f2=real(ifft2(F1));
            f3=fftshift(f2);

            p1r=conv2(p1_,f3,'same');
            p2r=conv2(p2_,f3,'same');
            p3r=conv2(p3_,f3,'same');

            p1_f=fftshift(fft2(p1r));
            p2_f=fftshift(fft2(p2r));
            p3_f=fftshift(fft2(p3r));

            aa=20*log(abs(p1_f));
            b=20*log(abs(p2_f));
            c=20*log(abs(p3_f));

            [X,Y] = meshgrid(-127:128);

            h=figure, subplot(331), mesh(f3/max(f3(:))); 
                    colormap(jet), xlabel('s_x'), ylabel('s_y'),
                    title('MTF filter[Spatial Domain]')
                    shading interp
                    freezeColors

                    subplot(334), imagesc(p1r),
                    colormap( gray(128)*diag([1,0,0])), axis off, axis square,
                    title({'\color{red}MTF*R Channel'});
                    freezeColors

                    subplot(335), imagesc(p2r),
                    colormap( gray(128)*diag([0,1,0])), axis off, axis square,
                    title({'\color{green}MTF*G Channel'});
                    freezeColors
                    
                    subplot(336), imagesc(p3r),
                    colormap( gray(128)*diag([0,0,1])), axis off, axis square,
                    title({'\color{blue}MTF*B Channel'});
                    freezeColors

                    subplot(337), mesh(X,Y, aa/max(aa(:))),
                    xlabel('f_x'), ylabel('f_y'), zlabel('amplitudes'),
                    colormap( gray(128)*diag([1,0,0])), 
                    freezeColors

                    subplot(338), mesh(X,Y, b/max(b(:))),
                    xlabel('f_x'), ylabel('f_y'), zlabel('amplitudes'),
                    colormap( gray(128)*diag([0,1,0])),
                    freezeColors


                    subplot(339), mesh(X,Y, c/max(c(:))),
                    xlabel('f_x'), ylabel('f_y'), zlabel('amplitudes'),
                    colormap( gray(128)*diag([0,0,1])),
                    freezeColors
                            
            suptitle('Phase II MTF')
            set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 1, 0.96]);
            saveas(h,sprintf('FIG12%d.png',i));
            pause(1)
            close all;

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Low_Pass Filters build with butterfilter for Noise(See LGN function)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            butter_filter=constructbutterfilter(size(p1r,1),20,5);
            [xx,yy] = calccpfov(size(p1r,1));
            radius = sqrt(xx.^2+yy.^2);  % cycles per field-of-view

            p1rr = imagefilter(p1r, butter_filter);
            p2rr = imagefilter(p2r, butter_filter);
            p3rr = imagefilter(p3r, butter_filter);

            cc=20*log(abs(fftshift(fft2(butter_filter))));
            cc=cc/max(cc(:));

            h=figure;
            subplot(331); mesh(xx,yy, cc);
            colormap(jet);xlabel('f_x'), ylabel('f_y'), zlabel('amplitudes');
            title('Low-pass filter(Fourier Domain)');
            freezeColors;

            subplot(334), imagesc(p1rr);
            colormap( gray(128)*diag([1,0,0])), axis off, axis square, 
            title({'\color{red}filter(MTF*R Channel)'});
            freezeColors

            subplot(335), imagesc(p2rr);
            colormap( gray(128)*diag([0,1,0])), axis off, axis square, 
            title({'\color{green}filter(MTF*G Channel)'});
            freezeColors

            subplot(336), imagesc(p3rr);
            colormap( gray(128)*diag([0,0,1])), axis off, axis square,
            title({'\color{blue}filter(MTF*B Channel)'});
            freezeColors

            qq=20*log(abs(fftshift(fft2(p1rr))));
            qq=qq/max(qq(:));

            subplot(337), mesh(xx,yy,qq);
            colormap( gray(128)*diag([1,0,0])),
            xlabel('f_x'), ylabel('f_y'), zlabel('amplitudes'),
            title({'\color{red}filter(MTF*R Channel)'});
            freezeColors

            rr=20*log(abs(fftshift(fft2(p2rr))));
            rr=rr/max(rr(:));

            subplot(338), mesh(xx,yy,rr);
            colormap( gray(128)*diag([0,1,0])), 
            xlabel('f_x'), ylabel('f_y'), zlabel('amplitudes')
            title({'\color{green}filter(MTF*G Channel)'});
            freezeColors

            hh=20*log(abs(fftshift(fft2(p3rr))));
            hh=hh/max(hh(:));

            subplot(339), mesh(xx,yy,hh);
            colormap( gray(128)*diag([0,0,1])), 
            xlabel('f_x'), ylabel('f_y'), zlabel('amplitudes')
            title({'\color{blue}filter(MTF*B Channel)'});
            freezeColors
            suptitle('Phase II Filter*(MTF)')
            set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 1, 0.96]);
            saveas(h,sprintf('FIG13%d.png',i));
            pause(1)
            close all;

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %      Phase III:  Photoreceptor Relative Response                       %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            load SmithPokornyCones
            load displaySPD

            % Matrix from XYZ to LMS and XYZ to ATD (Ingling and Tsou) 
            LMS=xyz2con([1 0 0;0 1 0;0 0 1],5);
            Mxyz2lms = LMS';
            ATD=xyz2atd([1 0 0;0 1 0;0 0 1],5);
            Mxyz2atd = ATD';
            T_lms = Mxyz2lms*(T_l(:,2:4)');
            T_atd = Mxyz2atd*(T_l(:,2:4)');

            h=figure,plot(wavelength, displaySPD(:,1),'r', wavelength, displaySPD(:,2),'g', wavelength, displaySPD(:,3),'b')
            xlabel('wavelength(nm)'),
            ylabel('radiance')
            legend('Red phosphor','Green phosphor','Blue phosphor')
            title('Monitor spectral power distribution(SPD)')
            saveas(h,sprintf('FIG14%d.png',i));

            h=figure,plot(wavelength,cones(:,1),'r',wavelength,cones(:,2),'g', wavelength,cones(:,3),'b')
            xlabel('wavelength(nm)'),
            ylabel('relative response(normalized)')
            legend('L cone','M cone','S cone' )
            title('Cone sensitivity function')
            saveas(h,sprintf('FIG15%d.png',i));


            h=figure, plot(T_l(:,1),T_atd(1,:),'r-',T_l(:,1),T_atd(2,:),'g-',T_l(:,1),T_atd(3,:),'b-')
            xlabel('wavelength(nm)'),
            ylabel('relative response(normalized)')
            legend('A opponent','T opponent','D opponent' )
            title('ATD (Ingling and Tsou) response ')
            saveas(h,sprintf('FIG16%d.png',i));

            % Palette of digital counts to tristimulus values (gamma calibration)
            Txyz=val2tri(n,Yw,tm,a,g);

            rgb2lms = cones'* displaySPD;
            rgbWhite = [1 1 1];
            whitepoint = rgbWhite * rgb2lms';

            imgRGB2=[p1rr p2rr p3rr];
            img1LMS = changeColorSpace(imgRGB2,rgb2lms);
            [p4,p5,p6]=getPlanes(img1LMS);

            p4_f=fftshift(fft2(p4));
            p5_f=fftshift(fft2(p5));
            p6_f=fftshift(fft2(p6));

            h=figure;

            subplot(231), imagesc(p4);
            axis square,
            axis off;
            colormap( gray(128)*diag([1,0,0]));
            title({'\color{red}L cone response'});
            freezeColors

            subplot(232), imagesc(p5);
            axis square,
            axis off;
            colormap( gray(128)*diag([0,1,0]));
            title({'\color{green}M cone response'});
            freezeColors

            subplot(233), imagesc(p6);
            axis square,
            axis off;
            colormap( gray(128)*diag([0,0,1]));
            title({'\color{blue}S cone response'});
            freezeColors

            ui=20*log(abs(p4_f));
            ui=ui/max(ui(:));
            subplot(234), mesh(xx,yy,ui),
            xlabel('f_x'), ylabel('f_y'), zlabel('amplitudes'),
            colormap( gray(128)*diag([1,0,0])), 
            freezeColors


            uo=20*log(abs(p5_f));
            uo=uo/max(uo(:));
            subplot(235), mesh(xx, yy, uo),
            xlabel('f_x'), ylabel('f_y'), zlabel('amplitudes'),
            colormap( gray(128)*diag([0,1,0])),
            freezeColors


            uf=20*log(abs(p6_f));
            uf=uf/max(uf(:));
            subplot(236), mesh(uf),
            xlabel('f_x'), ylabel('f_y'), zlabel('amplitudes'),
            colormap( gray(128)*diag([0,0,1])),
            freezeColors
                            
            suptitle('Phase II  Photoreceptor Relative Response L:M:S=12:6:1')
            set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 1, 0.96]);
            saveas(h,sprintf('FIG17%d.png',i));

            imageformat = 'lms';
            imageformat = [imageformat '   '];
            imageformat = imageformat(1:5);
            pause(1)
            close all;

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Phase III LMS to Chromatic Opponent Channels (Poirson&Wandell opponent) (RG-YB-WB) 
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            writeline('***********************************************');
            writeline('***********************************************');
            writeline('Apply Von-Kries in LMS -Chromatic Adaptation');
            writeline('***********************************************');
            writeline('***********************************************');

            % Convert XYZ or LMS representation to Poirson&Wandell opponent 1995, Vision Research
            % representation.
            fprintf('\nChromatic appearance model,\nFrom Paper,\nPattern—color separable pathways predict sensitivity to simple colored patterns\n');

            if (imageformat=='xyz10' | imageformat=='lms10')
              xyztype = 10;
            else
              xyztype = 2;
            end
            [L M S]=getPlanes(img1LMS);
            img1LMS=[L M S];
            if (imageformat(1:3)=='lms')
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


            h=figure, subplot(231), 
                    imagesc(LMSV(:,:,1)), 
                    axis square;
                    axis off;
                    colormap( gray(128)*diag([1,0,0]));
                    title({'\color{red}L adaption signal'});
                    freezeColors

                    subplot(232), 
                    imagesc(LMSV(:,:,2)), 
                    axis square;
                    axis off;
                    colormap( gray(128)*diag([0,1,0]));
                    title({'\color{green}M adaption signal'});
                    freezeColors

                    subplot(233), 
                    imagesc(LMSV(:,:,3)), 
                    axis square;
                    axis off;
                    colormap( gray(128)*diag([0,0,1]));
                    title({'\color{blue}S adaption signal'});
                    freezeColors

                    ppo=20*log(abs(p7_f));
                    ppo=ppo/max(ppo(:));
                    subplot(234), mesh(xx,yy,ppo),
                    xlabel('f_x'), ylabel('f_y'), zlabel('amplitudes'),
                    colormap( gray(128)*diag([1,0,0])), 
                    freezeColors

                    pp1=20*log(abs(p8_f));
                    pp1=pp1/max(pp1(:));
                    subplot(235), mesh(xx,yy,pp1),
                    xlabel('f_x'), ylabel('f_y'), zlabel('amplitudes'),
                    colormap( gray(128)*diag([0,1,0])),
                    freezeColors

                    pp2=20*log(abs(p9_f));
                    pp2=pp2/max(pp2(:));
                    subplot(236), mesh(xx,yy,pp2),
                    xlabel('f_x'), ylabel('f_y'), zlabel('amplitudes'),
                    colormap( gray(128)*diag([0,0,1])),
                    freezeColors

            suptitle('Phase III  Chromatic Adaptation Response with Von Kries Model')
            set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 1, 0.96]);
            saveas(h,sprintf('FIG18%d.png',i));

            pause(1)
            close all;

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Achromatic=L+M : adapting liminiance
            % Chromatic= L-M
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            L_adapt=LMSV(:,:,1)+LMSV(:,:,2);
            LMSV_updated=[LMSV(:,:,1) LMSV(:,:,2) LMSV(:,:,3)];
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Visual opponent 
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %Now, two type Visual Opponent 
            
            %ATD='InglingTsouOpponent';
            AATTDD=inputdlg('Enter a number:');
            
            switch AATTDD{1}

                case '1'
                    
                    % Matrix from XYZ to LMS and XYZ to ATD  
                    Tatd = Mxyz2atd*Txyz';
                    T_achromatic = [Tatd(1,:);0*Tatd(1,:);0*Tatd(1,:)];
                    T_T = [mean(Tatd(1,:))*ones(size(Tatd(1,:)));Tatd(2,:);0*Tatd(1,:)];
                    T_D = [mean(Tatd(1,:))*ones(size(Tatd(1,:)));0*Tatd(2,:);Tatd(3,:)];
                    imATD = pal2true(im_index,Tatd');

                    % Transform the colors from XYZ to ATD
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %Add RG-YB into visual  opponent
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    Txyz_a = inv(Mxyz2atd)*T_achromatic;
                    Txyz_t = inv(Mxyz2atd)*T_T;
                    Txyz_d = inv(Mxyz2atd)*T_D;


                    [nA,saturat,Tn]=tri2val(Txyz_a',Yw,tm,a,g,8);
                    [nT,saturat,Tn]=tri2val(Txyz_t',Yw,tm,a,g,8);
                    [nD,saturat,Tn]=tri2val(Txyz_d',Yw,tm,a,g,8);

                    imA = pal2true(im_index,nA);
                    imT = pal2true(im_index,nT);
                    imD = pal2true(im_index,nD);

                    h=figure,
                    subplot(131),image(imA),title('A')
                    subplot(132),image(imT),title('T')
                    subplot(133),image(imD),title('D')
                    saveas(h,sprintf('FIG19%d.png',i));


                    imAa=[imA(:,:,1) imA(:,:,2) imA(:,:,3)];
                    imTt=[imT(:,:,1) imT(:,:,2) imT(:,:,3)];
                    imDd=[imD(:,:,1) imD(:,:,2) imD(:,:,3)];

                    YAn_f=fftshift(fft2(imAa));
                    YDn_f=fftshift(fft2(imTt));
                    YTn_f=fftshift(fft2(imDd));

                    YAn_f1=20*log(abs(YAn_f));
                    YDn_f1=20*log(abs(YDn_f));
                    YTn_f1=20*log(abs(YTn_f));


                    h=figure, subplot(231), 
                            imagesc(imA), 
                            axis square;
                            axis off;
                            title('Luminance WB');

                            subplot(232), 
                            imagesc(imT), 
                            axis square;
                            axis off;
                            title('Opponent RG');

                            subplot(233), 
                            imagesc(imD), 
                            axis square;
                            axis off;
                            title('Opponent YB');

                            subplot(234), mesh(YAn_f1/max(YAn_f1(:))), colormap(jet),
                            xlabel('f_x'), ylabel('f_y'), zlabel('amplitudes'),
                            
                            subplot(235), mesh(YDn_f1/max(YDn_f1(:))),colormap(jet),
                            xlabel('f_x'), ylabel('f_y'), zlabel('amplitudes'),
                            
                            subplot(236), mesh(YTn_f1/max(YTn_f1(:))),colormap(jet),
                            xlabel('f_x'), ylabel('f_y'), zlabel('amplitudes'),
                        
                    suptitle('Phase VI Ingling and Tsou Opponent')
                    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 1, 0.96]);
                    saveas(h,sprintf('FIG20%d.png',i));
                    pause(1)
                    close all;

                case '2'

                    img1XaYaZa = changeColorSpace(LMSV_updated, cmatrix('lms2xyz', xyztype));
                    opp2 = changeColorSpace(img1XaYaZa, cmatrix('xyz2opp', xyztype));
            
                    
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %%  Prepare filters %%
                    %%  Separate pattern and color %%
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    disp('Preparing filters ...');

                    [k1, k2, k3] = separableFilters(sampPerDeg, 2);
                    
                    k1_f=fftshift(fft2(k1));
                    k2_f=fftshift(fft2(k2));
                    k3_f=fftshift(fft2(k3));


                    h=figure, subplot(231), mesh(k1),
                            axis square,
                            axis off,
                            colormap(jet),
                            title('Luminance WBfilter') 
                            freezeColors

                            subplot(232), mesh(k2),
                            axis square,
                            axis off,
                            colormap(jet)
                            title('Opponent RG filter') 
                            freezeColors

                            subplot(233), mesh(k3),
                            axis square,
                            axis off,
                            title('Opponent YB filter')
                            colormap(jet)
                            freezeColors

                            subplot(234), mesh(20*log(abs(k1_f))),
                            axis square,
                            xlabel('f_x'), ylabel('f_y'), zlabel('amplitudes'),
                            colormap(gray)
                            freezeColors

                            subplot(235), mesh(20*log(abs(k2_f))),
                            axis square,
                            xlabel('f_x'), ylabel('f_y'), zlabel('amplitudes'),
                            colormap(spring)
                            freezeColors

                            subplot(236), mesh(20*log(abs(k3_f))),
                            axis square,
                            xlabel('f_x'), ylabel('f_y'), zlabel('amplitudes'),
                            colormap(winter)
                            freezeColors
                    suptitle('Phase VII Opponent Spatial Filters ')
                    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 1, 0.96]);
                    saveas(h,sprintf('FIG19%d.png',i));

                    pause(1)
                    close all;
    
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %%  Spatial Filtering 
                    %%  The Pattern-Color Separable Model,1993 Brian Wandell Stanford University
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % Appearance of colored patterns: pattern–color separability, 1993 

                    % Apply the filters k1, k2, k3 to the images.
                    % The edges of the images are reflected for convolution.

                    [w1, w2, w3] = getPlanes(opp2);

                    wsize = size(w1);

                    disp('Filtering BW plane of image1 ...');
                    p10 = separableConv(w1, k1, abs(k1));
                    disp('Filtering RG plane of image1 ...');
                    p11 = separableConv(w2, k2, abs(k2));
                    disp('Filtering BY plane of image1 ...');
                    p12 = separableConv(w3, k3, abs(k3));

                    p10_f=fftshift(fft2(p10));
                    p11_f=fftshift(fft2(p11));
                    p12_f=fftshift(fft2(p12));

                    p10_f11=20*log(abs(p10_f));
                    p11_f11=20*log(abs(p11_f));
                    p12_f11=20*log(abs(p12_f));


                    h=figure, subplot(231), 
                            imagesc(p10),
                            colormap(gray), 
                            axis square;
                            axis off;
                            title('Luminance WB');
                            freezeColors

                            subplot(232), 
                            imagesc(p11),
                            colormap(gray), 
                            axis square;
                            axis off;
                            title('Opponent RG');
                            freezeColors

                            subplot(233), 
                            imagesc(p12),
                            colormap(gray), 
                            axis square;
                            axis off;
                            title('Opponent YB');
                            freezeColors


                            subplot(234), mesh(p10_f11/max(p10_f11(:))), colormap(jet),
                            xlabel('f_x'), ylabel('f_y'), zlabel('amplitudes'),
                            freezeColors
                            
                            subplot(235), mesh(p11_f11/max(p11_f11(:))),colormap(jet),
                            xlabel('f_x'), ylabel('f_y'), zlabel('amplitudes'),
                            freezeColors
                            
                            subplot(236), mesh(p12_f11/max(p12_f11(:))),colormap(jet),
                            xlabel('f_x'), ylabel('f_y'), zlabel('amplitudes'),
                            freezeColors

                    suptitle('Phase VII Pattern–Color separability')
                    set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 1, 0.96]);
                    saveas(h,sprintf('FIG20%d.png',i));

                    pause(1)
                    close all;

                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % Divisive Normalization Model 
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %Check third Version
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %                                   LGN --INFORMATION COMPRESSED                                       %
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %RECEPTIVE FIELDS
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                    % Discrete domain (2 degrees)
                    N = 64;
                    fs = 64;
                    [x,y,t,fx,fy,ft] = spatio_temp_freq_domain(N,N,1,fs,fs,1);

                    % Reference CSF
                    [h,CSSFO,CSFT,OE]=csfsso(fs,N,330.74,7.28,0.837,1.809,1,6.664);

                    CSFTn = CSFT/max(CSFT(:));

                    % LGN cell 
                    %
                    % I start from the default parameters from the 2D example in the help of the 
                    % function and I look for the best fit with the CSF imposing zero mean.
                    %

                    sigmax_pp = linspace(0.005,0.1,20);
                    sigmax_nn = linspace(0.006,1,100);

                    columns_x = N;
                    rows_y = N;
                    frames_t = 1;    
                    fsx = fs;
                    fsy = fs;
                    fst = 24;
                    xo_p = max(x(:))/2;
                    yo_p = max(x(:))/2;
                    to_p = 0.3;
                    order_p = 1;
                    sigmax_p = 0.02;
                    sigmat_p = 0.1;
                    xo_n = max(x(:))/2;
                    yo_n = max(x(:))/2;
                    to_n = 0.3;
                    order_n = 1;
                    sigmax_n = 0.2;
                    sigmat_n = 0.1;
                    excit_vs_inhib = 1;

                    for i=1:length(sigmax_pp)
                        for j=1:length(sigmax_nn)
                            
                            sigmax_p = sigmax_pp(i);
                            sigmax_n = sigmax_nn(j);
                            
                            [G,G_excit,G_inhib] = sens_lgn3d_space(columns_x,rows_y,frames_t,fsx,fsy,fst,xo_p,yo_p,to_p,order_p,sigmax_p,sigmat_p,xo_n,yo_n,to_n,order_n,sigmax_n,sigmat_n,excit_vs_inhib);
                            G = G_excit/sum(G_excit(:)) - G_inhib/sum(G_inhib(:));
                            
                            F = abs(fftshift(fft2(G)));
                            F = F/max(F(:));

                            departure=log(sum((F(:)-CSFTn(:)).^2));
                            error(i,j) = log(sum((F(:)-CSFTn(:)).^2)) + 4*(sigmax_p.^2 + sigmax_n.^2);
                            % this error function minimizes departure (first term)
                            % and wiring-length (second term). relative weight of
                            % both terms was set by hand
                            
                            indice_i(i,j) = i;
                            indice_j(i,j) = j;
                            % [i j]
                        end
                    end

                    [m,p]=min(error(:));
                    indi=indice_i(:);
                    indj=indice_j(:);

                    sigmax_p = sigmax_pp(indi(p));
                    sigmax_n = sigmax_nn(indj(p));

                    [G,G_excit,G_inhib] = sens_lgn3d_space(columns_x,rows_y,frames_t,fsx,fsy,fst,xo_p,yo_p,to_p,order_p,sigmax_p,sigmat_p,xo_n,yo_n,to_n,order_n,sigmax_n,sigmat_n,excit_vs_inhib);
                    G = G_excit/sum(G_excit(:)) - G_inhib/sum(G_inhib(:));

                    h=figure,mesh(x(1,:),y(:,1),G),title('Representative LGN receptive field')
                    xlabel('x (deg)'),ylabel('y (deg)')
                    saveas(h,sprintf('FIG21%d.png',i));

                    F = abs(fftshift(fft2(G)));
                    h=figure,mesh(fx(1,:),fy(:,1),F),title('Band-pass function of the LGN')
                    xlabel('f_x (cycl/deg)'),ylabel('f_y (cycl/deg)')
                    saveas(h,sprintf('FIG22%d.png',i));

                    h=figure,mesh(sigmax_nn,sigmax_pp,error), title('error function minimizes departure');
                    saveas(h,sprintf('FIG23%d.png',i));


                    h=figure(155),

                    subplot(2,2,1), mesh(x(1,:),y(:,1),G),
                    xlabel('x (deg)'),ylabel('y (deg)'),
                    title('LGN receptive field');

                    subplot(2,2,2), mesh(fx(1,:),fy(:,1),F),
                    xlabel('f_x (cycl/deg)'),ylabel('f_y (cycl/deg)'),
                    title('Band-pass function');

                    hold on,

                    % 2. Application of a system of receptive fields(only achromatic)
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % Flow into LGN with p10
                    % Image (4 degrees)
                    N = 256;
                    fs = 64;
                    [x,y,t,fx,fy,ft] = spatio_temp_freq_domain(N,N,1,fs,fs,1);

                    % 2.1 Explicit application: convolution
                    % -------------------------------------

                    xx1 = x(1,1:4:end); % Set of locations
                    yy1 = x(1,1:4:end);

                    columns_x = N;
                    rows_y = N;
                    frames_t = 1;      % Only one frame!
                    fsx = fs;
                    fsy = fs;
                    fst = 24;
                    to_p = 0.3;
                    order_p = 1;
                    sigmax_p = 0.03;  % Optimum we found above
                    sigmat_p = 0.1;
                    to_n = 0.3;
                    order_n = 1;
                    sigmax_n = 0.21;
                    sigmat_n = 0.1;
                    excit_vs_inhib = 1;

                    resp = zeros(length(xx1),length(xx1));
                    for i=1:length(xx1)
                        for j=1:length(yy1)

                            xo_p = xx1(i);
                            yo_p = yy1(j);
                            xo_n = xx1(i);
                            yo_n = yy1(j);
                            
                            [G,G_excit,G_inhib] = sens_lgn3d_space(columns_x,rows_y,frames_t,fsx,fsy,fst,xo_p,yo_p,to_p,order_p,sigmax_p,sigmat_p,xo_n,yo_n,to_n,order_n,sigmax_n,sigmat_n,excit_vs_inhib);
                            G = G_excit/sum(G_excit(:)) - G_inhib/sum(G_inhib(:));

                            resp(j,i) = G(:)'*p10(:);
                            
                        end
                        i
                    end

                    % figure,colormap gray,viewimage(xx,xx,resp),title('LGN Response')
                    % xlabel('x (deg)'),ylabel('y (deg)')

                    Resp = fftshift(abs(fft2(resp)));

                    subplot(2,2,3); imagesc(xx1,xx1,resp); colormap gray;
                    axis equal tight; xlabel('x (deg)'), ylabel('y (deg)'); 
                    title('LGN Response');
                    subplot(2,2,4); imagesc(xx1,xx1,Resp); colormap gray; colorbar; axis equal tight; 
                    xlabel('x (deg)'), ylabel('y (deg)');
                    title('magnitude spectrum');
                    saveas(h,sprintf('FIG24%d.png',i));
                    %print('-depsc',[figurepath,  strcat(['lgn', num2str(i)]), '.eps']);
                    

                    close all;
                    h=figure;

                    setfigurepos([50 50 800 500]);
                    subplot(2,3,1); imagesc(xx1,xx1,resp); colormap gray; colorbar; axis equal tight; xlabel('x (deg)'), ylabel('y (deg)'); title('LGN Response');
                    subplot(2,3,2); plot(resp(round((end+1)/2),:),'r.-'); ax = axis; axis([1 size(resp,2) ax(3:4)]); title('central row of response');
                    subplot(2,3,3); plot(resp(:,round((end+1)/2)),'r.-'); ax = axis; axis([1 size(resp,1) ax(3:4)]); title('central column of response');
                    subplot(2,3,4); imagesc(xx1,xx1,Resp); colormap gray; colorbar; axis equal tight; xlabel('x (deg)'), ylabel('y (deg)'); title('magnitude spectrum');
                    subplot(2,3,5); plot(Resp(round((end+1)/2),:),'r.-'); ax = axis; axis([1 size(resp,2) ax(3:4)]); title('central row of magnitude spectrum');
                    subplot(2,3,6); plot(Resp(:,round((end+1)/2)),'r.-'); ax = axis; axis([1 size(resp,1) ax(3:4)]); title('central column of magnitude spectrum');
                    drawnow;
                    saveas(h,sprintf('FIG25%d.png',i));

                    pause(1)
                    close all;

                    % 2.2 Application in the Fourier domain
                    %----------------------------------------

                    % Locate the receptive field in the center of the domain to avoid edge effects
                    xo_p = xx1(end)/2;
                    yo_p = yy1(end)/2;
                    xo_n = xx1(end)/2;
                    yo_n = yy1(end)/2;
                    [G,G_excit,G_inhib] = sens_lgn3d_space(columns_x,rows_y,frames_t,fsx,fsy,fst,xo_p,yo_p,to_p,order_p,sigmax_p,sigmat_p,xo_n,yo_n,to_n,order_n,sigmax_n,sigmat_n,excit_vs_inhib);
                    G = G_excit/sum(G_excit(:)) - G_inhib/sum(G_inhib(:));

                    % Compute the filter from the modulus (to avoid phase shifts due to the location of the RF away from the origin)
                    F = abs(fftshift(fft2(G)));
                    at = fftshift(fft2(p10));

                    B = real(ifft2(ifftshift(at.*F)));

                    % figure,colormap gray, viewimage(xx,xx,B),title('LGN Response (applic. of filter in the Fourier domain)')
                    % xlabel('x (deg)'),ylabel('y (deg)')
                    Resp_B = fftshift(abs(fft2(B)));
                    h=figure;

                    setfigurepos([50 50 800 500]);
                    subplot(2,3,1); imagesc(xx1,xx1,B); colormap gray; colorbar; axis equal tight; xlabel('x (deg)'), ylabel('y (deg)'); title('LGN Response');
                    subplot(2,3,2); plot(B(round((end+1)/2),:),'r.-'); ax = axis; axis([1 size(B,2) ax(3:4)]); title('central row of response');
                    subplot(2,3,3); plot(B(:,round((end+1)/2)),'r.-'); ax = axis; axis([1 size(B,1) ax(3:4)]); title('central column of response');
                    subplot(2,3,4); imagesc(xx1,xx1,Resp_B); colormap gray; colorbar; axis equal tight; xlabel('x (deg)'), ylabel('y (deg)'); title('magnitude spectrum');
                    subplot(2,3,5); plot(Resp_B(round((end+1)/2),:),'r.-'); ax = axis; axis([1 size(B,2) ax(3:4)]); title('central row of magnitude spectrum');
                    subplot(2,3,6); plot(Resp_B(:,round((end+1)/2)),'r.-'); ax = axis; axis([1 size(B,1) ax(3:4)]); title('central column of magnitude spectrum');
                    drawnow;
                    saveas(h,sprintf('FIG26%d.png',i));

                    pause(1)
                    close all;

                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % Primary Visual Cortex V1 Simulating with Energy model based on Gabor filters
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                    % S1 map: Simple V1 neurals
                    % C1 map: energy response  Complex V1 neurals
                    % Modified from from Serre et al. PAMI07 HMAX model
                    % Simulating the multi-scale && multi-orientational of V1
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                    % Filter size and normalization parameters may change according to different
                    % image resolution.
                    % Multi-Scale Multi-Resulation
                    Pscale=[2.8 3.6 4.5 5.4 6.3 7.3 8.2 9.2 10.2 11.3 12.3 13.4 14.6 15.8 17.0 18.2];          
                    Pwavelength=[3.5 4.6 5.6 6.8 7.9 9.1 10.3 11.5 12.7 14.1 15.4 16.8 18.2 19.7 21.2 22.8];   
                    Pfiltersize=[7:2:37]; 

                    % Image paramaters
                    shortestside=140;                                    % Images are rescaled so the shortest is equal to "shortestside" keeping aspect ratio

                    % Layer 1 parameters
                    NumbofOrient=12;                                    % Number of spatial orientations for the Gabor filter on the first layer 
                    Numberofscales=8;                                   % Number of scales for the Gabor filter on the first laye: must be between 1 and 16.
                                                                        % Modify line 7-9 of create_gabors.m to increase to more than than 16 scales
                    % Layer 2 layer parameters
                    maxneigh=floor(8:length(Pscale)/Numberofscales:8+length(Pscale));  % Size of maximum filters (if necessary adjust according Gabor filter sizes)
                    L2stepsize=4;                                                      % Step size of L2 max filter (downsampling)
                    inhi=0.5;                                                          % Local inhibition ceofficient (competition) between Gabor outputs at each position.
                                                                                       % Coefficient is between 0 and 1                                                  % Modify line 7-9 of create_gabors.m to increase to more than than 16 scales

                    %% LOAD AND DISPLAY GABOR FILTERS
                    Gabor=create_gabors(Numberofscales,NumbofOrient,Pscale,Pwavelength,Pfiltersize);
                    displaygabors(Gabor)
                    h=figure,
                    displayFFTgabors(Gabor)
                    saveas(h,sprintf('FIG27%d.png',i));
                    h=figure,
                    displayMeshFFTgabors(Gabor)
                    saveas(h,sprintf('FIG28%d.png',i));

                    pause(1)
                    close all;

                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %  V1 Simple Neural Response  ------ S1
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

                    % Two Dimensional
                    % More Example see ad_hoc_functions in MiNi_V1_Cortex
                    % The specific of properties of V1 can be simulating by Gabor function
                    % More example see Mini_V1_Cortex folder


                    writeline('------------------------------------------------')
                    writeline('------------------------------------------------')
                    writeline('------------------------------------------------')
                    writeline('Distribution of Spatial and Chromatic Information')
                    writeline('------------------------------------------------')
                    writeline('------------------------------------------------')
                    writeline('------------------------------------------------')


                    spat_chrom=imread('Images/spatial-chroma.png'), imagesc(spat_chrom); axis off; title('Adapted from Jesus Malo(2020)')

                    %Combine Color info(p11,p12) and Spatial info(B)
                    V1_flow_in=[B p11 p12]; %2dim
                    % V1_flow_in=cat(3, B, p11, p12); %3dim

                    % L1 LAYER  (NORMALIZED DOT PRODUCT OF GABORS FILTERS ON LOCAL PATCHES OF IMAGE "A" AT EVERY POSSIBLE LOCATIONS AND SCALES)
                    % The simple V1 cells response to spatial and chromatic channels
                    L1 = L1_layer(Gabor, V1_flow_in);

                    h=figure,
                    ca=0;
                    for (i=1:size(Gabor,1))
                        for (j=1:size(Gabor,2))
                            ca=ca+1;
                            subplot(size(Gabor,1),size(Gabor,2),ca)
                            imagesc(L1(:,:,ca)); axis off; colormap(gray);
                        end
                    end
                    suptitle('simple V1 cells response to spatial and chromatic channels')
                    print('-depsc',[figurepath,  strcat(['v1simple', num2str(i)]), '.eps']);
                    saveas(h,sprintf('FIG29%d.png',i));
                    

                    h=figure,
                    ca=0;
                    for (i=1:size(Gabor,1))
                        for (j=1:size(Gabor,2))
                            ca=ca+1;
                            subplot(size(Gabor,1),size(Gabor,2),ca)
                            imagesc(20*log(abs(fftshift(fft2(L1(:,:,ca)))))); axis off; colormap(jet);
                        end
                    end
                    suptitle('simple V1 cells response to spatial and chromatic channels [Fourier Domain]')
                    saveas(h,sprintf('FIG30%d.png',i));


                    h=figure,
                    ca=0;
                    for (i=1:size(Gabor,1))
                        for (j=1:size(Gabor,2))
                            ca=ca+1;
                            subplot(size(Gabor,1),size(Gabor,2),ca)
                            mesh(20*log(abs(fftshift(fft2(L1(:,:,ca)))))); axis off; colormap(jet);
                        end
                    end
                    suptitle('simple V1 cells response to spatial and chromatic channels [3D Fourier Domain]')
                    saveas(h,sprintf('FIG31%d.png',i));

                    close all;
                    % L2 LAYER: LOCAL MAX POOLING OF L1 LAYER OVER LOCAL POSITION AT ALL SCALES AND ALL ORIENTATIONS
                    % THE MAXIMUM POOLING SLIDING WINDOW SIZES ARE CONTAINED IN "maxneigh" AND "L2stepsize" INDICATES THE CORRESPONDING STEPSIZE 
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % The classic energy model for complex cells. The response of a complex cell is modelledby linearly filtering with quadrature-phase
                    % Gabor filters (Gabor functions whose sinusoidal com-ponents have a 90 degrees phase difference), taking squares, and summing. 
                    % Note that this is purelya mathematical description of the response and should not bedirectly interpreted as a hierarchicalmodel 
                    %summing simple cell responses.
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    fprintf('\nModel adapted from The book of Natural Image Statistics(2009)\n');
                    energy_model_principle=imread('Images/Energy_Model.png'); imagesc(energy_model_principle); axis off; title('Energy Model-Complex Cells');

                    L2 = L2_layer(L1,L2stepsize,maxneigh);

                    h=figure,
                    ca=0;
                    for (i=1:(size(Gabor,1)-1))
                        for (j=1:size(Gabor,2))
                            ca=ca+1;
                            subplot((size(Gabor,1))-1,size(Gabor,2),ca)
                            imagesc(L2(:,:,ca)); axis off; colormap(gray);
                        end
                    end
                    suptitle('complex V1 cells response to spatial and chromatic channels [Spatial Domain]')
                    %print('-depsc',[figurepath,  strcat(['v1complex', num2str(i)]), '.eps']);
                    saveas(h,sprintf('FIG32%d.png',i));
                    

                    h=figure,
                    ca=0;
                    for (i=1:(size(Gabor,1)-1))
                        for (j=1:size(Gabor,2))
                            ca=ca+1;
                            subplot((size(Gabor,1))-1,size(Gabor,2),ca)
                            imagesc(20*log(abs(fftshift(fft2(L2(:,:,ca)))))); axis off; colormap(jet);
                        end
                    end
                    suptitle('complex V1 cells response to spatial and chromatic channels [Fourier Domain]')
                    saveas(h,sprintf('FIG33%d.png',i));



                    h=figure,
                    ca=0;
                    for (i=1:(size(Gabor,1)-1))
                        for (j=1:size(Gabor,2))
                            ca=ca+1;
                            subplot((size(Gabor,1))-1,size(Gabor,2),ca)
                            mesh(20*log(abs(fftshift(fft2(L2(:,:,ca)))))); axis off; colormap(jet);
                        end
                    end
                    suptitle('complex V1 cells response to spatial and chromatic channels [3D Fourier Domain]')
                    saveas(h,sprintf('FIG34%d.png',i));

                    close all;

                    disp('''''''''''''''''''''''''''''''''''''''''''''''''');
                    disp('''''''''''''''''''''''''''''''''''''''''''''''''');
                    disp('V1 mainly respose spatila information than chromatic information(70:30)');
                    disp('''''''''''''''''''''''''''''''''''''''''''''''''');
                    disp('''''''''''''''''''''''''''''''''''''''''''''''''');

                    % Now we use 17 filters to reconstructure the V1 propertie of Multi-Scale, Multi-Orientation. 
                    % Very similarity steerable wavelet transform which dependent matlabPyrTools.

                    load sensors_V1_64x64_3

                    % Here, We only care spatial infmation
                    [rs,resp,p,ind_gran] = apply_V1_matrix_to_blocks_256_image(B,ggs,indices,0.7,1);

                    % =================================
                    % Multi-channel decomposition
                    % nCSF based 
                    % Wavelets Transform with steerable pyramid 
                    % =================================

                    writeline('------------------------------------')
                    writeline('Wavelets Transform with steerable pyramid')
                    writeline('Multi-channel decomposition')
                    writeline('------------------------------------')

                    V1_flow_in=cat(3, B, p11, p12);

                    V1_flow_in=[V1_flow_in(:,:,1), V1_flow_in(:,:,2), V1_flow_in(:,:,3)];
                    [bands.pyr,bands.pind] = buildSpyr(V1_flow_in, 'auto', 'sp3Filters' );

                    %plot the transform result from above
                    h=figure,
                    showSpyr(bands.pyr, bands.pind)
                    % 70% spatial information and 30% chromatic information
                    title('V1 More Strongly Selectivity Spatial information than chromation information')
                    %print('-depsc',[figurepath,  strcat(['v1Spatchrom', num2str(i)]), '.eps']);
                    saveas(h,sprintf('FIG35%d.png',i));
                    
                    close all;
                    % Here we only focus on Spatial (achromatic information)
                    [v1_1 v1_2 v1_3]=getPlanes(V1_flow_in);
                    [bands.pyr,bands.pind] = buildSpyr(v1_1, 'auto', 'sp3Filters' );
                    %plot the transform result from above
                    h=figure,
                    showSpyr(bands.pyr, bands.pind)
                    title('Spatial channel')
                    %print('-depsc',[figurepath,  strcat(['v1Spatial', num2str(i)]), '.eps']);
                    saveas(h,sprintf('FIG36%d.png',i));
                    
                    close all;

                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    % Visualization All Reuslt
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %Retina
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    
                    hFig = figure(111);

                    subplot(4,7,1), imagesc(img);
                    axis square;
                    axis off;
                    colormap('default');
                    title('Stimuli');
                    freezeColors

                    subplot(4,7,2), imagesc(img(:,:,1));
                    axis square;
                    axis off;
                    colormap(gray);
                    title({'\color{red}R channel'});
                    freezeColors 

                    subplot(4,7,3), imagesc(img(:,:,2));
                    axis square;
                    axis off;
                    colormap(gray);
                    title({'\color{green}G channel'});
                    freezeColors 

                    subplot(4,7,4), imagesc(img(:,:,3));
                    axis square;
                    axis off;
                    colormap(gray);
                    title({'\color{blue}B channel'});
                    freezeColors 

                    subplot(4,7,5), imagesc(p1);
                    axis square;
                    axis off;
                    title({'\color{red}Calibr R channel'});
                    colormap(gray);
                    freezeColors 

                    subplot(4,7,6), imagesc(p2);
                    axis square;
                    axis off;
                    colormap(gray);
                    title({'\color{green}Calibr G channel'});
                    freezeColors 

                    subplot(4,7,7), imagesc(p3);
                    axis square;
                    axis off;
                    colormap(gray);
                    title({'\color{blue}Calibr B channel'});
                    freezeColors

                    subplot(4,7,8), imagesc(Imd);
                    axis square;
                    axis off;
                    colormap( gray);
                    title(['Gaussian White Noise  \sigma^2=',num2str(variance)])
                    freezeColors

                    subplot(4,7, 9), imagesc(p1_),
                    colormap(gray), axis off, axis square,
                    title({'\color{red}R Channel+noise'});
                    freezeColors

                    subplot(4,7, 10), imagesc(p2_),
                    colormap(gray), axis off, axis square,
                    title({'\color{green}G Channel+noise'});
                    freezeColors
                            
                    subplot(4,7, 11), imagesc(p3_),
                    colormap( gray), axis off, axis square,
                    title({'\color{blue}B Channel+noise'});
                    freezeColors

                
                    subplot(4,7, 12), imagesc(p1r),
                    colormap( gray), axis off, axis square,
                    title({'\color{red}MTF*R Channel'});
                    freezeColors

                    subplot(4,7, 13), imagesc(p2r),
                    colormap( gray), axis off, axis square,
                    title({'\color{green}MTF*G Channel'});
                    freezeColors
                            
                    subplot(4,7, 14), imagesc(p3r),
                    colormap( gray), axis off, axis square,
                    title({'\color{blue}MTF*B Channel'});
                    freezeColors

                    subplot(4,7, 15); mesh(xx,yy, cc);
                    colormap(jet);xlabel('f_x'), ylabel('f_y'), zlabel('amplitudes');
                    title('Low-pass filter(Fourier Domain)');
                    freezeColors;


                    subplot(4,7, 16), imagesc(p1rr);
                    colormap( gray), axis off, axis square, 
                    title({'\color{red}filter(MTF*R Channel)'});
                    freezeColors

                    subplot(4,7, 17), imagesc(p2rr);
                    colormap( gray), axis off, axis square, 
                    title({'\color{green}filter(MTF*G Channel)'});
                    freezeColors

                    subplot(4,7, 18), imagesc(p3rr);
                    colormap( gray), axis off, axis square,
                    title({'\color{blue}filter(MTF*B Channel)'});
                    freezeColors

                    subplot(4,7, 19), imagesc(p4);
                    axis square,
                    axis off;
                    colormap( gray);
                    title({'\color{red}L cone response'});
                    freezeColors

                    subplot(4,7, 20), imagesc(p5);
                    axis square,
                    axis off;
                    colormap(gray);
                    title({'\color{green}M cone response'});
                    freezeColors

                    subplot(4,7, 21), imagesc(p6);
                    axis square,
                    axis off;
                    colormap(gray);
                    title({'\color{blue}S cone response'});
                    freezeColors

                    subplot(4,7, 23), imagesc(LMSV(:,:,1)), 
                    axis square;
                    axis off;
                    colormap(gray);
                    title({'\color{red}L adaption signal'});
                    freezeColors

                    subplot(4,7, 24), imagesc(LMSV(:,:,2)), 
                    axis square;
                    axis off;
                    colormap(gray);
                    title({'\color{green}M adaption signal'});
                    freezeColors
                            
                    subplot(4,7, 25), imagesc(LMSV(:,:,3)), 
                    axis square;
                    axis off;
                    colormap(gray);
                    title({'\color{blue}S adaption signal'});
                    freezeColors

                    subplot(4,7, 26), imagesc(p10),
                    colormap(gray), 
                    axis square;
                    axis off;
                    title('Luminance WB');
                    freezeColors

                    subplot(4,7, 27), imagesc(p11),
                    colormap(gray), 
                    axis square;
                    axis off;
                    title('Opponent RG');
                    freezeColors

                    subplot(4,7, 28), imagesc(p12),
                    colormap(gray), 
                    axis square;
                    axis off;
                    title('Opponent YB');
                    freezeColors

                    suptitle('Phase I Retina Processing')
                    set(hFig, 'Units', 'Normalized', 'OuterPosition', [0, 0.04, 1, 0.96]);
                    saveas(hFig,sprintf('FIG37%d.png',i));

                    %print('-depsc',[figurepath,  strcat(['retina', num2str(i)]), '.eps']);

                    close all;
                    clearvars -except color_data Msx T_l Yw tm g a figurepath;
                    
                    !mv *.png  Images/Result/
                
                otherwise
                    disp('Not found any visual opponent model!!!!'); 
            end
        end   


    case 'ICAminiCortex'

        disp('Taking long time and go to externel and check LGN-v1 scripts!!!')

    case 'spectraldataset'
    
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        disp('*******************************************************************************');
        disp('*******************************************************************************');
        disp('-----------------Spectral Dataset Demo LINK Spectral_ATD_Color.m---------------');
        disp('*******************************************************************************');
        disp('*******************************************************************************');
        % Here just load can't be used in this script!!!!
        % Spectral datasets download from https://github.com/isetbio/isetbio/wiki/ISETBIO-Data
        load Images/Scene13.mat
        N = 40;
        [up,y,l]=size(S)
        im_radiance_N = zeros(N,N,length(Wavelengths));
        for i=1:length(Wavelengths)
            im_radiance_N(:,:,i) = 350*imresize(squeeze(S(:,:,i)),[N N]);  % Check lambda resampling in spec2tri
        end    
        % wavelength 401-700
        % Visualization
        for i=1:l
            figure(1), imagesc(S(:,:,i)); axis off;
            pause(0.001) % This command freezes figure(1) for 0.05sec. 
        end
        slice = S(:,:,17);
        figure; imagesc(slice); colormap('gray'); brighten(0.5);
        z = max(slice(100, 39));
        slice_clip = min(slice, z)/z;
        figure; imagesc(slice_clip.^0.4); colormap('gray');
        reflectance = squeeze(S(141, 75,:));
        figure; plot(Wavelengths, reflectance, 'r-');
        xlabel('wavelength, nm');
        ylabel('unnormalized reflectance'); 

        % Database from https://personalpages.manchester.ac.uk/staff/d.h.foster/Hyperspectral_images_of_natural_scenes_02.html
        load Images/Foster2002_HyperImgDatabase/scene1.mat
        N = 40;
        [up,y,l]=size(reflectances)
        im_radiance_N = zeros(N,N,31);
        for i=1:31
            im_radiance_N(:,:,i) = 350*imresize(squeeze(reflectances(:,:,i)),[N N]);  % Check lambda resampling in spec2tri
        end    
        % wavelength 400:10:700
        % Visualization
        for i=1:l
            figure(1), imagesc(reflectances(:,:,i)); axis off;
            pause(0.001) % This command freezes figure(1) for 0.05sec. 
        end
        slice = reflectances(:,:,17);
        figure; imagesc(slice); colormap('gray'); brighten(0.5);
        z = max(slice(100, 39));
        slice_clip = min(slice, z)/z;
        figure; imagesc(slice_clip.^0.4); colormap('gray');
        reflectance = squeeze(reflectances(141, 75,:));
        figure; plot(400:10:700, reflectance, 'r-');
        xlabel('wavelength, nm');
        ylabel('unnormalized reflectance'); 


    otherwise
            warndlg('GO BACK CHECK LOOP!!! SOMETHING WRONG!!!');
    
end