%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                                 Spectral-Retina-LMS-ATD(Ingling & Tsou Visual Opponent Model)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Dataset: FosterDatabase
% Paper: Time-lapse ratios of cone excitations in natural scenes (2016)

addpath(genpath('/home/qiang/QiangLi/Matlab_Utils_Functional/matlab_human-vision-model_utils/Retina-LGN-V1-models-OnGoing/ForstDatabased'));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Dependent Colorlab toolbox
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
startcol

% Ingling & Tsou Model(Chrom   atic Opponent Model) with Smith & Pokorny fundamentals
Mxyz2atd = (xyz2atd(eye(3),5,[],2))';
Matd2xyz = inv(Mxyz2atd);      
Mxyz2lms = xyz2con(eye(3),5)';
Mlms2xyz = inv(Mxyz2lms);

Tatd = (Mxyz2atd*T_l(:,2:4)')';
Tlms = (Mxyz2lms*T_l(:,2:4)')';

Mlms2atd = Mxyz2atd*inv(Mxyz2lms);
Matd2lms = inv(Mlms2atd);

figure,plot(T_l(:,1),T_l(:,2:4)),title('Color Matching Functions XYZ')
figure,plot(T_l(:,1),Tatd),title('Color Matching Functions ATD (Ingling)')
figure,plot(T_l(:,1),Tlms),title('Color Matching Functions LMS (Smith & Pokorny)')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% lambdas meas wavelength from 400-700, Step is 10;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lambdas = 400:10:720;


% Scene Category Have 4:levada,gualtar,nogueiro,sete_fontes   
for scene = 1:4
    
    if scene==1
        nom=['levada_1411';'levada_1518';'levada_1608';'levada_1712';'levada_1810';'levada_1830';'levada_1845'];
        pos_W_rows = 405:410;
        pos_W_cols = 15:20;
        name = 'levada';
    elseif scene ==2
        nom=['gualtar_1144';'gualtar_1245';'gualtar_1346';'gualtar_1447';'gualtar_1545';'gualtar_1645';'gualtar_1746';'gualtar_1853';'gualtar_1944'];
        pos_W_rows = 240:245;
        pos_W_cols = 295:300;
        name = 'gualtar';
    elseif scene ==3
        nom=['nogueiro_1140';'nogueiro_1240';'nogueiro_1345';'nogueiro_1441';'nogueiro_1600';'nogueiro_1637';'nogueiro_1745';'nogueiro_1845';'nogueiro_1941'];
        pos_W_rows = 320:325;
        pos_W_cols = 420:425;
        name = 'nogueiro';
    elseif scene ==4
        nom=['sete_fontes_1225';'sete_fontes_1321';'sete_fontes_1438';'sete_fontes_1515';'sete_fontes_1614';'sete_fontes_1713';'sete_fontes_1815';'sete_fontes_1840'];
        pos_W_rows = 490:495;
        pos_W_cols = 515:520;
        name = 'sete_fontes';
    end
    
    Nc = 250;
    factorvisual = 80;
    
    n_fot = size(nom,1);
    T_XYZ = zeros(Nc,3,n_fot);
    T_XYZv = zeros(Nc,3,n_fot);
    
    im_XYZ = zeros(512,672,3,n_fot);
    im_XYZ_v = zeros(512,672,3,n_fot);
    
    imind = zeros(512,672,n_fot);
    nn = zeros(Nc,3,n_fot);
    
    imindv = zeros(512,672,n_fot);
    nnv = zeros(Nc,3,n_fot);
    
    Wlms = zeros(3,n_fot);
    
    
    for nombre =1:n_fot
        
        load(['/media/disk/databases/BBDD_video_image/Image_Statistic/DataFoster/time_lapse/',nom(nombre,:),'.mat'])
        HSI = abs( hsi(1:2:end,1:2:end,:) );
        

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Step: SPECTRAL INTRGRATION ADT%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %  SPECTRAL INTRGRATION (colorlab functions) -> ima in XYZ
        %  AD-HOC SPECTRAL INTEGRATION (using LMS color matching functions computed above) -> ima in LMS
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        TLMS(:,1) = interp1(T_l(:,1),Tlms(:,1),lambdas);
        TLMS(:,2) = interp1(T_l(:,1),Tlms(:,2),lambdas);
        TLMS(:,3) = interp1(T_l(:,1),Tlms(:,3),lambdas);
        
        ima = HSI(:,:,1:3);
        for i=1:size(ima,1)
            for j=1:size(ima,2)
                T = TLMS'*squeeze(HSI(i,j,:));
                ima(i,j,:)= T;
            end
            i
        end
        ima_LMS = ima;
        
        imaATD(:,:,1) = Mlms2atd(1,1)*ima_LMS(:,:,1) + Mlms2atd(1,2)*ima_LMS(:,:,2) + Mlms2atd(1,3)*ima_LMS(:,:,3);
        imaATD(:,:,2) = Mlms2atd(2,1)*ima_LMS(:,:,1) + Mlms2atd(2,2)*ima_LMS(:,:,2) + Mlms2atd(2,3)*ima_LMS(:,:,3);
        imaATD(:,:,3) = Mlms2atd(3,1)*ima_LMS(:,:,1) + Mlms2atd(3,2)*ima_LMS(:,:,2) + Mlms2atd(3,3)*ima_LMS(:,:,3);
        
        factor = 0.001*randn(size(imaATD(:,:,1)));
        fact(:,:,1) = factor;fact(:,:,2) = factor;fact(:,:,3) = factor;
        imaATD = imaATD + fact.*imaATD;
        
        % Display XYZ image
        [im_ind,T_atd] = true2pal( imaATD ,Nc);
        T_xyz = (Matd2xyz*T_atd')';
        [n,saturat,Tn]=tri2val(factorvisual*T_xyz,Yw,tm,a,g,8);
        figure(n_fot+nombre),colormap(n),image(im_ind)
        figure(n_fot+nombre+1),colordgm(T_xyz,1,T_l,Yw,'symb','.','sizes(3)',8,'showtriang',{2,tm})
        if isempty(find(saturat==1))
        else
            figure(n_fot+nombre+1),hold on,colordgm(Tn(find(saturat==1),:),1,T_l,Yw,'symb','.','sizes(3)',8,'showtriang',{2,tm},'linecolors(7,:)',[1 0 0])
        end
        
        T_XYZ(:,:,nombre) = T_xyz;
        
        imaXYZ(:,:,1) = Mlms2xyz(1,1)*ima_LMS(:,:,1) + Mlms2xyz(1,2)*ima_LMS(:,:,2) + Mlms2xyz(1,3)*ima_LMS(:,:,3);
        imaXYZ(:,:,2) = Mlms2xyz(2,1)*ima_LMS(:,:,1) + Mlms2xyz(2,2)*ima_LMS(:,:,2) + Mlms2xyz(2,3)*ima_LMS(:,:,3);
        imaXYZ(:,:,3) = Mlms2xyz(3,1)*ima_LMS(:,:,1) + Mlms2xyz(3,2)*ima_LMS(:,:,2) + Mlms2xyz(3,3)*ima_LMS(:,:,3);
        
        im_XYZ(:,:,:,nombre) = imaXYZ;
        imind(:,:,nombre) = im_ind;
        nn(:,:,nombre) = n;
        
        %%
        %% VON-KRIES ADAPTATION in LMS
        %%
        
        Lo = squeeze(mean(mean(ima_LMS(pos_W_rows,pos_W_cols,:),1),2));
        
        Wlms(:,nombre) = Lo;
        
        % Lo_canonic = Mxyz2lms*[1 1 1]';
        Lo_canonic = Mxyz2lms*inv(Mxyz2atd)*[1 0 0]';
        
        ima_LMSv(:,:,1) = (Lo_canonic(1)/Lo(1))*ima_LMS(:,:,1);
        ima_LMSv(:,:,2) = (Lo_canonic(2)/Lo(2))*ima_LMS(:,:,2);
        ima_LMSv(:,:,3) = (Lo_canonic(3)/Lo(3))*ima_LMS(:,:,3);
        
        %%
        %% OPPONENT CHANNELS
        %%
        
        ima_ATD(:,:,1) = Mlms2atd(1,1)*ima_LMSv(:,:,1) + Mlms2atd(1,2)*ima_LMSv(:,:,2) + Mlms2atd(1,3)*ima_LMSv(:,:,3);
        ima_ATD(:,:,2) = Mlms2atd(2,1)*ima_LMSv(:,:,1) + Mlms2atd(2,2)*ima_LMSv(:,:,2) + Mlms2atd(2,3)*ima_LMSv(:,:,3);
        ima_ATD(:,:,3) = Mlms2atd(3,1)*ima_LMSv(:,:,1) + Mlms2atd(3,2)*ima_LMSv(:,:,2) + Mlms2atd(3,3)*ima_LMSv(:,:,3);
        
        %
        % Adapted image back in XYZ
        %
        
        ima_v(:,:,1) = Matd2xyz(1,1)*ima_ATD(:,:,1) + Matd2xyz(1,2)*ima_ATD(:,:,2) + Matd2xyz(1,3)*ima_ATD(:,:,3);
        ima_v(:,:,2) = Matd2xyz(2,1)*ima_ATD(:,:,1) + Matd2xyz(2,2)*ima_ATD(:,:,2) + Matd2xyz(2,3)*ima_ATD(:,:,3);
        ima_v(:,:,3) = Matd2xyz(3,1)*ima_ATD(:,:,1) + Matd2xyz(3,2)*ima_ATD(:,:,2) + Matd2xyz(3,3)*ima_ATD(:,:,3);
        
        % Display adapted XYZ image
        imafake = ima_ATD;
        imafake = imafake + fact.*imafake;
        [im_ind_v,T_atd_v] = true2pal( imafake ,Nc);
        T_xyz_v = (Matd2xyz*T_atd_v')';
        [nv,saturatv,Tnv] = tri2val(factorvisual*T_xyz_v,Yw,tm,a,g,8);
        figure(n_fot+nombre+2),colormap(nv),image(im_ind_v)
        figure(n_fot+nombre+3),colordgm(T_xyz_v,1,T_l,Yw,'symb','.','sizes(3)',8,'showtriang',{2,tm})
        if isempty(find(saturatv==1))
        else
            hold on,colordgm(Tnv(find(saturatv==1),:),1,T_l,Yw,'symb','.','sizes(3)',8,'showtriang',{2,tm},'linecolors(7,:)',[1 0 0])
        end
        im_XYZ_v(:,:,:,nombre) = ima_v;
        T_XYZ_v(:,:,nombre) = T_xyz_v;
        
        imindv(:,:,nombre) = im_ind_v;
        nnv(:,:,nombre) = nv;
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%
    cd /media/disk/vista/Papers/2017_Information_Flow/DataFoster/time_lapse
    save(['analysis_color_',name],'T_XYZ','T_XYZv','im_XYZ','im_XYZ_v','imind','nn','imindv','nnv','Wlms','n_fot')
    clear T_XYZ T_XYZv im_XYZ im_XYZ_v imind nn imindv nnv Wlms n_fot
    
end

% colores = zeros(n_fot,3);
% colores(:,1:2) = [linspace(0,0.9,n_fot)' linspace(0,0.9,n_fot)'];
% 
% coloresW = colores;
% coloresW(:,2:3) =  [linspace(0,0.9,n_fot)' linspace(0,0.9,n_fot)'];
% coloresW(end:-1:1,1) = linspace(0,1,n_fot);
% 
% colores=colores(end:-1:1,:);
% coloresW=coloresW(end:-1:1,:);
% 
% for i = 1:n_fot
%     lab = xyz2lab(T_XYZ(:,:,i),(Mlms2xyz*Wlms(:,i))');
%     figure(100) 
%     hold on,colordgm(T_XYZ(:,:,i),1,T_l,Yw,'symb','.','sizes(3)',8,'linecolors(7,:)',colores(i,:))
%     hold on,colordgm((Mlms2xyz*Wlms(:,i))',1,T_l,Yw,'symb','s','sizes(3)',8,'linecolors(7,:)',coloresW(i,:)),title('No adapt')
%     figure(101) 
%     hold on,colordgm(T_XYZ_v(:,:,i),1,T_l,Yw,'symb','.','sizes(3)',8,'linecolors(7,:)',colores(i,:)),title('Adapt')
%     figure(102),hold on,colorspc(T_XYZ(:,:,i),1,T_l,Yw,'showvectors',0,'symb','.','sizes(3)',8,'showtriang',{3,tm},'linecolors(8,:)',colores(i,:))
%                 hold on,colorspc((Mlms2xyz*Wlms(:,i))',1,T_l,Yw,'showvectors',1,'symb','.','sizes(3)',8,'showtriang',{3,tm},'linecolors(7:8,:)',[coloresW(i,:);coloresW(i,:)]),title('No adapt')
%     figure(103),hold on,colorspc(T_XYZ_v(:,:,i),1,T_l,Yw,'showvectors',0,'symb','.','sizes(3)',8,'showtriang',{3,tm},'linecolors(8,:)',colores(i,:)),title('Adapt')
%     figure(104),hold on,plot(lab(:,2),lab(:,3),'.','color',colores(i,:)),xlabel('a'),ylabel('b')
%     figure(105),hold on,plot3(lab(:,1),lab(:,2),lab(:,3),'.','color',colores(i,:)),xlabel('L'),ylabel('a'),zlabel('b')
%     
%     figure(106),hold on,plot3(nn(:,1,i),nn(:,2,i),nn(:,3,i),'.','color',colores(i,:)),xlabel('n1'),ylabel('n2'),zlabel('n3'),title('Digital Values (no adapt)')
%     figure(107),hold on,plot3(nnv(:,1,i),nnv(:,2,i),nnv(:,3,i),'.','color',colores(i,:)),xlabel('n1'),ylabel('n2'),zlabel('n3'),title('Digital Values (adapt)')
%     pause
% end


noms = {'Lillies_Closeup';'Lilly_Closeup';'Sameiro_Leaves';'Bom_Jesus_Ruin';'Sameiro_Glade';'Tibaes_Corridor';'Braga_Grafitti';...
        'Ruivaes_Ferns';'Tenoes_Wall_Closeup';'Tibaes_Garden_Entrance';'Yellow_Rose';'Sete_Fontes_Rock';'Sameiro_Branch';...
        'Sameiro_Trees';'gualtar_Steps';'Ruivaes_Fern';'Sameiro_Bark';'Gualtar_Columns';'Souto_Wood_Pile';...
        'Bom_Jesus_Bush';'Gualtar_Villa';'Tenoes_Wall';'Bom_Jesus_Red_flower';'Souto_Farm_Barn';'Bom_Jesus_Marigold';...
        'Tibaes_Garden';'Souto_Roof_Tiles'};

posicW = zeros(27,2);
for i=1:27
    i
    noms{i}
    load(noms{i}) 
    HSI = hsi(1:2:end,1:2:end,:);
    figure(1),colormap gray,imagesc(abs(HSI(:,:,10)).^0.3);
    lala = round(ginput(1));
    posicW(i,:) = [lala(2) lala(1)];
end
    
posicW =[142   105;
   423   263;
   338   247;
   364    49;
   476   207;
   396   204;
   447   580;
   365   169;
   370   337;
   451   334;
   153   343;
   314   484;
   310   330;
   257   562;
    56   615;
   441    52;
   384   237;
   113   427;
   350   153;
   114   197;
   297   464;
   259   364;
    77   589;
   457   515;
   382   288;
   446   511;
   455   459];

startcol
% Ingling & Tsou Model with Smith & Pokorny fundamentals
Mxyz2atd = (xyz2atd(eye(3),5,[],2))';
Matd2xyz = inv(Mxyz2atd);      
Mxyz2lms = xyz2con(eye(3),5)';
Mlms2xyz = inv(Mxyz2lms);

Tatd = (Mxyz2atd*T_l(:,2:4)')';
Tlms = (Mxyz2lms*T_l(:,2:4)')';

Mlms2atd = Mxyz2atd*inv(Mxyz2lms);
Matd2lms = inv(Mlms2atd);

figure,plot(T_l(:,1),T_l(:,2:4)),title('Color Matching Functions XYZ')
figure,plot(T_l(:,1),Tatd),title('Color Matching Functions ATD (Ingling)')
figure,plot(T_l(:,1),Tlms),title('Color Matching Functions LMS (Smith & Pokorny)')

lambdas = 400:10:720;

Nc = 250;
    T_XYZ = zeros(Nc,3,27);
    T_XYZv = zeros(Nc,3,27);
    
    im_XYZ = zeros(512,672,3,27);
    im_XYZ_v = zeros(512,672,3,27);
    
    imind = zeros(512,672,27);
    nn = zeros(Nc,3,27);
    
    imindv = zeros(512,672,27);
    nnv = zeros(Nc,3,27);
    
    Wlms = zeros(3,27);

figura = 1;
imagen = zeros(1024,1344,33);
for scene = 1:27
    
        nom = noms{scene};
        pos_W_rows = posicW(scene,1):posicW(scene,1)+4;
        pos_W_cols = posicW(scene,2):posicW(scene,2)+4;
        name = num2str(scene);
            
    
        factorvisual = 50;
    
        load(['media/disk/databases/BBDD_video_image/Image_Statistic/DataFoster/general/',nom,'.mat'])
        tam = size(hsi);
        imagen(1:tam(1),1:tam(2),:) = hsi;
        HSI = abs( imagen(1:2:end,1:2:end,:) );
        imagen = 0*imagen;clear hsi
        
        % %%
        % %% SPECTRAL INTRGRATION (colorlab functions) -> ima in XYZ
        % %%
        % ima = HSI(:,:,1:3);
        % for i=1:size(ima,1)
        %     %tic
        %     %for j=1:size(ima,2)
        %     %    [T,RR]=spec2tri(T_l,10,[lambdas' squeeze(HSI(i,:,:))']);
        %     %    ima(i,:,1) = T(:,1);
        %     %    ima(i,:,2) = T(:,2);
        %     %    ima(i,:,3) = T(:,3);
        %     %end
        %     for j=1:size(ima,2)
        %         [T,RR]=spec2tri(T_l,10,[lambdas' squeeze(HSI(i,j,:))]);
        %         ima(i,j,:)= T;
        %     end
        %     i
        %     %toc
        % end
        
        %%
        %% AD-HOC SPECTRAL INTEGRATION (using LMS color matching functions computed above) -> ima in LMS
        %%
        
        TLMS(:,1) = interp1(T_l(:,1),Tlms(:,1),lambdas);
        TLMS(:,2) = interp1(T_l(:,1),Tlms(:,2),lambdas);
        TLMS(:,3) = interp1(T_l(:,1),Tlms(:,3),lambdas);
        
        ima = HSI(:,:,1:3);
        for i=1:size(ima,1)
            for j=1:size(ima,2)
                T = TLMS'*squeeze(HSI(i,j,:));
                ima(i,j,:)= T;
            end
            i
        end
        ima_LMS = ima;
        
        imaATD(:,:,1) = Mlms2atd(1,1)*ima_LMS(:,:,1) + Mlms2atd(1,2)*ima_LMS(:,:,2) + Mlms2atd(1,3)*ima_LMS(:,:,3);
        imaATD(:,:,2) = Mlms2atd(2,1)*ima_LMS(:,:,1) + Mlms2atd(2,2)*ima_LMS(:,:,2) + Mlms2atd(2,3)*ima_LMS(:,:,3);
        imaATD(:,:,3) = Mlms2atd(3,1)*ima_LMS(:,:,1) + Mlms2atd(3,2)*ima_LMS(:,:,2) + Mlms2atd(3,3)*ima_LMS(:,:,3);
        
        factor = 0.0005*randn(size(imaATD(:,:,1)));
        fact(:,:,1) = factor;fact(:,:,2) = factor;fact(:,:,3) = factor;
        imaATD = imaATD + fact.*imaATD;
        
        % Display XYZ image
        [im_ind,T_atd] = true2pal( imaATD ,Nc);
        T_xyz = (Matd2xyz*T_atd')';
        [n,saturat,Tn]=tri2val(factorvisual*T_xyz,Yw,tm,a,g,8);
        figure(figura),colormap(n),image(im_ind),title(['Original ',name])
        figura = figura+1;
        figure(figura),colordgm(T_xyz,1,T_l,Yw,'symb','.','sizes(3)',8,'showtriang',{2,tm}),title(['Original ',name])
        if isempty(find(saturat==1))
        else
            figure(figura),hold on,colordgm(Tn(find(saturat==1),:),1,T_l,Yw,'symb','.','sizes(3)',8,'showtriang',{2,tm},'linecolors(7,:)',[1 0 0])            
        end
        figura=figura+1;
        T_XYZ(:,:,scene) = T_xyz;
        
        imaXYZ(:,:,1) = Mlms2xyz(1,1)*ima_LMS(:,:,1) + Mlms2xyz(1,2)*ima_LMS(:,:,2) + Mlms2xyz(1,3)*ima_LMS(:,:,3);
        imaXYZ(:,:,2) = Mlms2xyz(2,1)*ima_LMS(:,:,1) + Mlms2xyz(2,2)*ima_LMS(:,:,2) + Mlms2xyz(2,3)*ima_LMS(:,:,3);
        imaXYZ(:,:,3) = Mlms2xyz(3,1)*ima_LMS(:,:,1) + Mlms2xyz(3,2)*ima_LMS(:,:,2) + Mlms2xyz(3,3)*ima_LMS(:,:,3);
        
        im_XYZ(:,:,:,scene) = imaXYZ;
        imind(:,:,scene) = im_ind;
        nn(:,:,scene) = n;
        
        %%
        %% VON-KRIES ADAPTATION in LMS
        %%
        
        Lo = squeeze(mean(mean(ima_LMS(pos_W_rows,pos_W_cols,:),1),2));
        
        Wlms(:,scene) = Lo;
        
        % Lo_canonic = Mxyz2lms*[1 1 1]';
        Lo_canonic = Mxyz2lms*inv(Mxyz2atd)*[1 0 0]';
        
        ima_LMSv(:,:,1) = (Lo_canonic(1)/Lo(1))*ima_LMS(:,:,1);
        ima_LMSv(:,:,2) = (Lo_canonic(2)/Lo(2))*ima_LMS(:,:,2);
        ima_LMSv(:,:,3) = (Lo_canonic(3)/Lo(3))*ima_LMS(:,:,3);
        
        %%
        %% OPPONENT CHANNELS
        %%
        
        ima_ATD(:,:,1) = Mlms2atd(1,1)*ima_LMSv(:,:,1) + Mlms2atd(1,2)*ima_LMSv(:,:,2) + Mlms2atd(1,3)*ima_LMSv(:,:,3);
        ima_ATD(:,:,2) = Mlms2atd(2,1)*ima_LMSv(:,:,1) + Mlms2atd(2,2)*ima_LMSv(:,:,2) + Mlms2atd(2,3)*ima_LMSv(:,:,3);
        ima_ATD(:,:,3) = Mlms2atd(3,1)*ima_LMSv(:,:,1) + Mlms2atd(3,2)*ima_LMSv(:,:,2) + Mlms2atd(3,3)*ima_LMSv(:,:,3);
        
        %
        % Adapted image back in XYZ
        %
        
        ima_v(:,:,1) = Matd2xyz(1,1)*ima_ATD(:,:,1) + Matd2xyz(1,2)*ima_ATD(:,:,2) + Matd2xyz(1,3)*ima_ATD(:,:,3);
        ima_v(:,:,2) = Matd2xyz(2,1)*ima_ATD(:,:,1) + Matd2xyz(2,2)*ima_ATD(:,:,2) + Matd2xyz(2,3)*ima_ATD(:,:,3);
        ima_v(:,:,3) = Matd2xyz(3,1)*ima_ATD(:,:,1) + Matd2xyz(3,2)*ima_ATD(:,:,2) + Matd2xyz(3,3)*ima_ATD(:,:,3);
        
        % Display adapted XYZ image
        imafake = ima_ATD;
        imafake = imafake + fact.*imafake;
        [im_ind_v,T_atd_v] = true2pal( imafake ,Nc);
        T_xyz_v = (Matd2xyz*T_atd_v')';
        [nv,saturatv,Tnv] = tri2val(factorvisual*T_xyz_v,Yw,tm,a,g,8);
        figure(figura),colormap(nv),image(im_ind_v),title(['Adapted ',name])
        figura=figura+1;
        figure(figura),colordgm(T_xyz_v,1,T_l,Yw,'symb','.','sizes(3)',8,'showtriang',{2,tm}),title(['Adapted ',name])
        if isempty(find(saturatv==1))
        else
            hold on,colordgm(Tnv(find(saturatv==1),:),1,T_l,Yw,'symb','.','sizes(3)',8,'showtriang',{2,tm},'linecolors(7,:)',[1 0 0])
        end
        figura = figura+1;
        im_XYZ_v(:,:,:,scene) = ima_v;
        T_XYZ_v(:,:,scene) = T_xyz_v;
        
        imindv(:,:,scene) = im_ind_v;
        nnv(:,:,scene) = nv;
  
    %%%%%%%%%%%%%%%%%%%%%%
    cd /media/disk/vista/Papers/2017_Information_Flow/DataFoster/general
    save(['analysis_color_general'],'T_XYZ','T_XYZ_v','im_XYZ','im_XYZ_v','imind','nn','imindv','nnv','Wlms')
        
end