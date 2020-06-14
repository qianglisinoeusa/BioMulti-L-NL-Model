function visual_pathway_third_version()


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                    Digtal Human Visual Cortex Pathway                                             %
%                                       Retina-------LGN---------V1                                                 %
%                                         linear + nonlinear model                                                  %
%                                                                                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% visual pathway simulated from retina - LGN - V1

clc;
clear all;
close all;


disp('***************************************************************************');
disp('***************************************************************************');
disp('***************************************************************************');
disp('                    ... retina-------LGN---------V1 ...                    ')
disp('***************************************************************************');
disp('***************************************************************************');
disp('***************************************************************************');


warning ('off');
startcol
cd ('/home/qiang/QiangLi/Matlab_Utils_Functional/matlab_human-vision-model_utils/Retina-LGN-V1-models-OnGoing');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Parameters;
load SmithPokornyCones;
load displaySPD;
load displayGamma;


params.ppd=0.0847;
params.NM=15;
params.DIS=500;
params.N = 256;
params.variance=0.005;
params.sampPerDeg=120;

fs=64;
columns_x = params.N;
rows_y = params.N;
frames_t = 1;      
fsx = fs;
fsy = fs;
fst = 24;
to_p = 0.3;
order_p = 1;
sigmax_p = 0.03;  
sigmat_p = 0.1;
to_n = 0.3;
order_n = 1;
sigmax_n = 0.21;
sigmat_n = 0.1;
excit_vs_inhib = 1;


% Matrix from XYZ to LMS and XYZ to ATD (Ingling and Tsou) 
LMS=xyz2con([1 0 0;0 1 0;0 0 1],5);
Mxyz2lms = LMS';
ATD=xyz2atd([1 0 0;0 1 0;0 0 1],5);
Mxyz2atd = ATD';
T_lms = Mxyz2lms*(T_l(:,2:4)');
T_atd = Mxyz2atd*(T_l(:,2:4)');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%mModel with Image
datatype='natureimage';

switch lower(datatype)
    
    case 'natureimage'
  
        data.folder= 'Images/Miscellaneous-USC-DataBase/misc/color_image';
        data.type      = 'png';
        files = dir(fullfile(data.folder,sprintf('*.%s',data.type)));
        color_data = {};
        for i = 1:length(files)
            fprintf('Processing image %04d of %04d\n',i,length(files));   
            img = imresize(imread(fullfile(data.folder,files(i).name)), [params.N, params.N]);
            img = im2double(img);
            [h,w,c] = size(img);
            imsize = size(img);
            if (imsize(1)>1 && prod(imsize(2:length(imsize)))>3)  
              dimension = 2;
            else
              dimension = 1;
            end
            samples = 1:(h*w);
            timg = reshape(img,[h*w c]);
            color_data{i} = timg(samples,:);
        end
        
        for i=1:size(color_data,2)
            img_raw=reshape(color_data{1,1}, params.N, params.N, 3);

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%For loop;
            img=img_raw;
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Gamma Correction;
            img1 = [img(:,:,1)  img(:,:,2)  img(:,:,3)];
            imgRGB1 = dac2rgb(img1, gammaTable);
            
            [p1,p2,p3]=getPlanes(imgRGB1);
            %imgRGB1_updated=cat(3,p1,p2,p3);
            
            
            h=figure;

            subplot(331), imagesc(img);
            axis equal;
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
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Retina Model;
            
            
            % Step 1:  Constructure Noise
            
            deg_model='wt_gn';
           
            [Imd, PSF] = KeCoDe_degradation(imgRGB1, deg_model,params.variance);
            
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

            
            % Step 2: MTF

            f=MTF(params.NM, params.ppd, params.DIS);

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
            
            % Step 3: Low_Pass Filters Noise
            
            butter_filter=constructbutterfilter(size(p1r,1),20,5);
            [xx,yy] = calccpfov(size(p1r,1));
            
            % cycles per field-of-view
            radius = sqrt(xx.^2+yy.^2);  
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
            
            % Achromatic=L+M : adapting liminiance
            % Chromatic= L-M
            L_adapt=LMSV(:,:,1)+LMSV(:,:,2);
            LMSV_updated=[LMSV(:,:,1) LMSV(:,:,2) LMSV(:,:,3)];
            
            % Step 6: Visual Opponent Channel

            img1XaYaZa = changeColorSpace(LMSV_updated, cmatrix('lms2xyz', xyztype));
            opp2 = changeColorSpace(img1XaYaZa, cmatrix('xyz2opp', xyztype));

            % Step 7: Opponent Filter
            
            [k1, k2, k3] = separableFilters(params.sampPerDeg, 2);
                    
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
                
                %Step 8: Nonlinear(Divisive Normalization)
        
                dn_img = MSCN(p10);  % WB channel 
                p11 = MSCN(p11); 
                p12= MSCN(p12); 
               
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%LGN Model;
                
                [x,y,t,fx,fy,ft] = spatio_temp_freq_domain(params.N, params.N,1,fs,fs,1);

                xx1 = x(1,1:4:end); 
                yy1 = x(1,1:4:end);
                
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
                
                B = MSCN(B);
                
                Resp_B = fftshift(abs(fft2(B)));
                
                subplot(1,2,1); imagesc(xx1,xx1,B); colormap gray;
                axis equal tight; xlabel('x (deg)'), ylabel('y (deg)'); 
                title('LGN Response');
                subplot(1,2,2); imagesc(xx1,xx1,Resp_B); colormap gray; colorbar; axis equal tight; 
                xlabel('x (deg)'), ylabel('y (deg)');
                title('magnitude spectrum');
                saveas(h,sprintf('FIG24%d.png',i));

                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%v1 Model;
                
                load sensors_V1_64x64_3
                % Here, We only care spatial infmation
                [rs,resp,p,ind_gran] = apply_V1_matrix_to_blocks_256_image(B,ggs,indices,0.7,1);
                
                V1_flow_in=cat(3, B, p11, p12);

                V1_flow_in=[V1_flow_in(:,:,1), V1_flow_in(:,:,2), V1_flow_in(:,:,3)];
                [bands.pyr,bands.pind] = buildSpyr(V1_flow_in, 'auto', 'sp3Filters' );
                
                h=figure,
                showSpyr(bands.pyr, bands.pind)
                % 70% spatial information and 30% chromatic information
                title('V1 More Strongly Selectivity Spatial information than chromation information')
                saveas(h,sprintf('FIG35%d.png',i));         
            
            
        end

end

end



function [MSCN_img]= MSCN(img)
    %Psychology named Divisive Normalization
    window = fspecial('gaussian',7,7/6); 
    window = window/sum(sum(window));
    mu = filter2(window, img, 'same');
    mu_sq = mu.*mu;
    sigma = sqrt(abs(filter2(window, ...
    img.*img, 'same') - mu_sq));
    imgMinusMu = (img-mu);
    MSCN_img =imgMinusMu./(sigma +1);

end