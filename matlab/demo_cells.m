function demo_cells(folder)

%%
%% DEMO for LGN, V1 and MT cells
%%

%% ILLUSTRATIVE STIMULUS 75 pix, 64 frames, fsx=37.5, fst = 20 ------------
% folder = 'C:\disco_portable\mundo_irreal\rutinas\Software_Katedra\Vistalab\sample_data'
load real_seq_Nx_75_Nf_64.mat  % sequence Y
  FTY = fft3( Y , 1);
  show_fft3( abs(FTY).^0.25,75,20, 4),axis square
  set(4,'color',[1 1 1])  
  view_fourier = [-113 24];
  view(view_fourier)
  
  Yn = (Y-min(Y(:)))/(max(Y(:))-min(Y(:)));
  Yn = then2now(Yn,75);
  
  implay(Yn)

%% LGN------------------------------------------------

  % Domain and cell parameters
            columns_x = 75;
            rows_y = 75;
            frames_t = 64;
            fsx = 37.5;
            fsy = 37.5;
            fst = 20;
            xo_p = 1;
            yo_p = 1;
            to_p = 0.5; 
            order_p = 1;
            sigmax_p = 0.05; 
            sigmat_p = 0.1; 
            xo_n = 1;
            yo_n = 1;
            to_n = 0.5; 
            order_n = 1;
            sigmax_n = 0.3; 
            sigmat_n = 0.1;
            excit_vs_inhib = 1;
 
  % Cell          
  [G,G_excit,G_inhib] = sens_lgn3d_space(columns_x,rows_y,frames_t,fsx,fsy,fst,xo_p,yo_p,to_p,order_p,sigmax_p,sigmat_p,xo_n,yo_n,to_n,order_n,sigmax_n,sigmat_n,excit_vs_inhib);
  
  % Domain
  [x,y,t,fx,fy,ft] = spatio_temp_freq_domain(rows_y,columns_x,frames_t,fsx,fsy,fst);

  % Normalization and 3D array
  factor = 0.3;
  GG = G_excit - factor*G_inhib;
  GG = GG/norm(GG);
  GG = (GG-min(GG(:)))/(max(GG(:))-min(GG(:)));
  GG3 = then2now(GG,columns_x);
  
  implay(GG3,fst)
 
  name = 'lgn_movie1';
  MOV = build_achrom_movie_avi(GG3,min(GG3(:)),max(GG3(:)),columns_x,1,fst,1,folder,name);
  
  name = 'lgn_movie2'; 
  vidObj = VideoWriter([folder,name,'.avi'],'Motion JPEG AVI');
  vidObj.FrameRate=fst;
  open(vidObj);
  expo = 0.8;
  for i=1:64
      figure(2),
      colormap([0 0 1;0 0 1])
      mesh(x(1,1:2:75),y(1:2:end,1),sign(GG3(1:2:end,1:2:end,i)-0.5).*(abs(GG3(1:2:end,1:2:end,i)-0.5)).^expo)
      xlabel('x (deg)'),ylabel('y (deg)')
      set(2,'color',[1 1 1])
      axis([0 2 0 2 -0.5.^expo 0.5.^expo]),axis square
      view([-13 15])
      lala = getframe(2);
      writeVideo(vidObj,lala);
end
close(vidObj);
  
  % Spectrum
  FTGG = fft3( GG , 1);
  show_fft3( abs(FTGG).^0.125, fsx, fst, 3),axis square
  set(3,'color',[1 1 1])  
  
  % Undo shift to the center (proper spectrum)
  FTGG0 = FTGG.*exp(2*pi*i*(1*fx+1*fy+0.5*ft));
  GG0 = ifft3( FTGG0 , 1);
  % GG0 = real(GG0);
  % GG0 = (GG0-min(GG0(:)))/(max(GG0(:))-min(GG0(:)));
  % GG03 = then2now(GG0,columns_x);
  % implay(GG03,fst)
  
%% Apply LGN to stimulus

YY = abs(ifft3(FTY.*abs(FTGG),1));
YYn = (YY-min(YY(:)))/(max(YY(:))-min(YY(:)));
YYn = then2now(YYn,75);
  show_fft3( (abs(FTY).*abs(FTGG)).^(0.09),75,20, 5),axis square
  set(5,'color',[1 1 1])  
  view_fourier = [-113 24];
  view(view_fourier)

  set(3,'color',[1 1 1])  
  view_fourier = [-113 24];
  view(view_fourier)
  
implay(YYn)

Y_lgn = cat(2,Yn,YYn);

name = 'lgn_movie3';
MOV = build_achrom_movie_avi(Y_lgn,min(Y_lgn(:)),max(Y_lgn(:)),columns_x,6,fst,1,folder,name);

%%%%%%%%%%%%%
%%%%%%%%%%%%%
%%%%%%%%%%%%%

%% V1 Sensors

% 
% In this sequence the interesting frequencies are:
%          fx  fy  ft 
%    f1 = [0  7.5  2  ];
%    f2 = [0  7.5  7.5];
%    f3 = [0  7.5  10 ];
%    f4 = [0  7.5  -2  ];
%    f5 = [0  7.5  -7.5];
%    f6 = [0  7.5  -10 ];
% 
% And sensible widths are:
%
%    delta_fx = 2
%    delta_fy = 2
%    delta_ft = 1.5

fxo=[0 0 0 0 0 0];
fyo=[7 7 7 7 7 7];
fto=[2 7 9 -2 -7 -9];
delta_fx=2;
delta_fy=2;
delta_ft=1.5;

G1 = sens_gabor3d_freq(columns_x,rows_y,frames_t,fsx,fsx,fst,fxo(1),fyo(1),fto(1),delta_fx,delta_fy,delta_ft);
G2 = sens_gabor3d_freq(columns_x,rows_y,frames_t,fsx,fsx,fst,fxo(2),fyo(2),fto(2),delta_fx,delta_fy,delta_ft);
G3 = sens_gabor3d_freq(columns_x,rows_y,frames_t,fsx,fsx,fst,fxo(3),fyo(3),fto(3),delta_fx,delta_fy,delta_ft);
G4 = sens_gabor3d_freq(columns_x,rows_y,frames_t,fsx,fsx,fst,fxo(4),fyo(4),fto(4),delta_fx,delta_fy,delta_ft);
G5 = sens_gabor3d_freq(columns_x,rows_y,frames_t,fsx,fsx,fst,fxo(5),fyo(5),fto(5),delta_fx,delta_fy,delta_ft);
G6 = sens_gabor3d_freq(columns_x,rows_y,frames_t,fsx,fsx,fst,fxo(6),fyo(6),fto(6),delta_fx,delta_fy,delta_ft);

i=sqrt(-1);
G1 = G1.*exp(2*pi*i*(1*fx+1*fy+0.5*ft));
G2 = G2.*exp(2*pi*i*(1*fx+1*fy+0.5*ft));
G3 = G3.*exp(2*pi*i*(1*fx+1*fy+0.5*ft));
G4 = G4.*exp(2*pi*i*(1*fx+1*fy+0.5*ft));
G5 = G5.*exp(2*pi*i*(1*fx+1*fy+0.5*ft));
G6 = G6.*exp(2*pi*i*(1*fx+1*fy+0.5*ft));

g1 = real(ifft3(G1,1));
g2 = real(ifft3(G2,1));
g3 = real(ifft3(G3,1));
g4 = real(ifft3(G4,1));
g5 = real(ifft3(G5,1));
g6 = real(ifft3(G6,1));

g1 = then2now((g1-min(g1(:)))/(max(g1(:))-min(g1(:))),75);
g2 = then2now((g2-min(g2(:)))/(max(g2(:))-min(g2(:))),75);
g3 = then2now((g3-min(g3(:)))/(max(g3(:))-min(g3(:))),75);
g4 = then2now((g4-min(g4(:)))/(max(g4(:))-min(g4(:))),75);
g5 = then2now((g5-min(g5(:)))/(max(g5(:))-min(g5(:))),75);
g6 = then2now((g6-min(g6(:)))/(max(g6(:))-min(g6(:))),75);

C1 = cat(2,g1,g2,g3);
C2 = cat(2,g4,g5,g6);
C = cat(1,C1,C2);

implay(C)

name = 'v1_movie1';
MOV = build_achrom_movie_avi(C,min(C(:)),max(C(:)),columns_x*3,20,fst,1,folder,name);

  name = 'v1_movie2'; 
  vidObj = VideoWriter([folder,name,'.avi'],'Motion JPEG AVI');
  vidObj.FrameRate=fst;
  open(vidObj);
  expo = 1;
  for i=1:64
      figure(8),
      subplot(231)
      GG3=g1;
      colormap([0 0 1;0 0 1])
      mesh(x(1,1:2:75),y(1:2:end,1),sign(GG3(1:2:end,1:2:end,i)-0.5).*(abs(GG3(1:2:end,1:2:end,i)-0.5)).^expo)
      xlabel('x (deg)'),ylabel('y (deg)')
      axis([0 2 0 2 -0.5.^expo 0.5.^expo]),axis square
      view([-75 15])
      subplot(232)
      GG3=g2;
      colormap([0 0 1;0 0 1])
      mesh(x(1,1:2:75),y(1:2:end,1),sign(GG3(1:2:end,1:2:end,i)-0.5).*(abs(GG3(1:2:end,1:2:end,i)-0.5)).^expo)
      xlabel('x (deg)'),ylabel('y (deg)')
      axis([0 2 0 2 -0.5.^expo 0.5.^expo]),axis square
      view([-75 15])
      subplot(233)
      GG3=g3;
      colormap([0 0 1;0 0 1])
      mesh(x(1,1:2:75),y(1:2:end,1),sign(GG3(1:2:end,1:2:end,i)-0.5).*(abs(GG3(1:2:end,1:2:end,i)-0.5)).^expo)
      xlabel('x (deg)'),ylabel('y (deg)')
      axis([0 2 0 2 -0.5.^expo 0.5.^expo]),axis square
      view([-75 15])
      subplot(234)
      GG3=g4;
      colormap([0 0 1;0 0 1])
      mesh(x(1,1:2:75),y(1:2:end,1),sign(GG3(1:2:end,1:2:end,i)-0.5).*(abs(GG3(1:2:end,1:2:end,i)-0.5)).^expo)
      xlabel('x (deg)'),ylabel('y (deg)')
      axis([0 2 0 2 -0.5.^expo 0.5.^expo]),axis square
      view([-75 15])
      subplot(235)
      GG3=g5;
      colormap([0 0 1;0 0 1])
      mesh(x(1,1:2:75),y(1:2:end,1),sign(GG3(1:2:end,1:2:end,i)-0.5).*(abs(GG3(1:2:end,1:2:end,i)-0.5)).^expo)
      xlabel('x (deg)'),ylabel('y (deg)')
      axis([0 2 0 2 -0.5.^expo 0.5.^expo]),axis square
      view([-75 15])
      subplot(236)
      GG3=g6;
      colormap([0 0 1;0 0 1])
      mesh(x(1,1:2:75),y(1:2:end,1),sign(GG3(1:2:end,1:2:end,i)-0.5).*(abs(GG3(1:2:end,1:2:end,i)-0.5)).^expo)
      xlabel('x (deg)'),ylabel('y (deg)')
      axis([0 2 0 2 -0.5.^expo 0.5.^expo]),axis square
      view([-75 15])
      set(8,'color',[1 1 1])
      lala = getframe(8);
      writeVideo(vidObj,lala);
end
close(vidObj);

figure(9), show_fft3(G1,fsx,fst,9),axis square,view(view_fourier),set(9,'color',[1 1 1])
figure(10),show_fft3(G2,fsx,fst,10),axis square,view(view_fourier),set(10,'color',[1 1 1])
figure(11),show_fft3(G3,fsx,fst,11),axis square,view(view_fourier),set(11,'color',[1 1 1])
figure(12),show_fft3(G4,fsx,fst,12),axis square,view(view_fourier),set(12,'color',[1 1 1])
figure(13),show_fft3(G5,fsx,fst,13),axis square,view(view_fourier),set(13,'color',[1 1 1])
figure(14),show_fft3(G6,fsx,fst,14),axis square,view(view_fourier),set(14,'color',[1 1 1])

%% Apply V1 to stimulus

Y1 = real(ifft3(FTY.*abs(G1),1));
Y2 = real(ifft3(FTY.*abs(G2),1));
Y3 = real(ifft3(FTY.*abs(G3),1));
Y4 = real(ifft3(FTY.*abs(G4),1));
Y5 = real(ifft3(FTY.*abs(G5),1));
Y6 = real(ifft3(FTY.*abs(G6),1));

Y1 = then2now(Y1,75);
Y2 = then2now(Y2,75);
Y3 = then2now(Y3,75);
Y4 = then2now(Y4,75);
Y5 = then2now(Y5,75);
Y6 = then2now(Y6,75);

m = min([Y1(:);Y2(:);Y3(:);Y4(:);Y5(:);Y6(:)]);
M = max([Y1(:);Y2(:);Y3(:);Y4(:);Y5(:);Y6(:)]);

Y1 = (Y1-m)/(M-m);
Y2 = (Y2-m)/(M-m);
Y3 = (Y3-m)/(M-m);
Y4 = (Y4-m)/(M-m);
Y5 = (Y5-m)/(M-m);
Y6 = (Y6-m)/(M-m);

C1 = cat(2,Yn,Y1,Y2,Y3);
C2 = cat(2,ones(75,75,64),Y4,Y5,Y6);
C = cat(1,C1,C2);

implay(C)

name = 'v1_movie3'; 
MOV = build_achrom_movie_avi(C,min(C(:)),max(C(:)),columns_x*4,15,fst,1,folder,name);

%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%

%% MT sensors -------------------------------------------------------

% Sensor's speed

   v1 = [0  0.1];
   v2 = [0  1];
   v3 = [0  1.7];
   v4 = [0  -0.1];
   v5 = [0  -1];
   v6 = [0  -1.7];

% Campos receptivos en el dominio de Fourier   
     
 G1 = sens_MT(columns_x,rows_y,frames_t,fsx,fsy,fst,v1);
 G2 = sens_MT(columns_x,rows_y,frames_t,fsx,fsy,fst,v2);
 G3 = sens_MT(columns_x,rows_y,frames_t,fsx,fsy,fst,v3);
 G4 = sens_MT(columns_x,rows_y,frames_t,fsx,fsy,fst,v4);
 G5 = sens_MT(columns_x,rows_y,frames_t,fsx,fsy,fst,v5);
 G6 = sens_MT(columns_x,rows_y,frames_t,fsx,fsy,fst,v6);
 
g1 = real(ifft3(G1.*exp(i*(2*pi*rand(75,75*64)-pi)),1));
g2 = real(ifft3(G2.*exp(i*(2*pi*rand(75,75*64)-pi)),1));
g3 = real(ifft3(G3.*exp(i*(2*pi*rand(75,75*64)-pi)),1));
g4 = real(ifft3(G4.*exp(i*(2*pi*rand(75,75*64)-pi)),1));
g5 = real(ifft3(G5.*exp(i*(2*pi*rand(75,75*64)-pi)),1));
g6 = real(ifft3(G6.*exp(i*(2*pi*rand(75,75*64)-pi)),1));

g1 = then2now((g1-min(g1(:)))/(max(g1(:))-min(g1(:))),75);
g2 = then2now((g2-min(g2(:)))/(max(g2(:))-min(g2(:))),75);
g3 = then2now((g3-min(g3(:)))/(max(g3(:))-min(g3(:))),75);
g4 = then2now((g4-min(g4(:)))/(max(g4(:))-min(g4(:))),75);
g5 = then2now((g5-min(g5(:)))/(max(g5(:))-min(g5(:))),75);
g6 = then2now((g6-min(g6(:)))/(max(g6(:))-min(g6(:))),75);

C1 = cat(2,g1,g2,g3);
C2 = cat(2,g4,g5,g6);
C = cat(1,C1,C2);

implay(C)

name = 'MT_movie1';
MOV = build_achrom_movie_avi(C,min(C(:)),max(C(:)),columns_x*3,16,fst,1,folder,name);

  name = 'MT_movie2'; 
  vidObj = VideoWriter([folder,name,'.avi'],'Motion JPEG AVI');
  vidObj.FrameRate=fst;
  open(vidObj);
  expo = 1;
  for i=1:64
      figure(8),
      subplot(231)
      GG3=g1;
      colormap([0 0 1;0 0 1])
      mesh(x(1,1:2:75),y(1:2:end,1),2*(0.25+sign(GG3(1:2:end,1:2:end,i)-0.5).*(abs(GG3(1:2:end,1:2:end,i)-0.5))).^expo)
      xlabel('x (deg)'),ylabel('y (deg)')
      axis([0 2 0 2 -0.5.^expo 0.5.^expo]),axis square
      view([-75 15])
      subplot(232)
      GG3=g2;
      colormap([0 0 1;0 0 1])
      mesh(x(1,1:2:75),y(1:2:end,1),2*(0.25+sign(GG3(1:2:end,1:2:end,i)-0.5).*(abs(GG3(1:2:end,1:2:end,i)-0.5))).^expo)
      xlabel('x (deg)'),ylabel('y (deg)')
      axis([0 2 0 2 -0.5.^expo 0.5.^expo]),axis square
      view([-75 15])
      subplot(233)
      GG3=g3;
      colormap([0 0 1;0 0 1])
      mesh(x(1,1:2:75),y(1:2:end,1),2*(0.25+sign(GG3(1:2:end,1:2:end,i)-0.5).*(abs(GG3(1:2:end,1:2:end,i)-0.5))).^expo)
      xlabel('x (deg)'),ylabel('y (deg)')
      axis([0 2 0 2 -0.5.^expo 0.5.^expo]),axis square
      view([-75 15])
      subplot(234)
      GG3=g4;
      colormap([0 0 1;0 0 1])
      mesh(x(1,1:2:75),y(1:2:end,1),2*(0.25+sign(GG3(1:2:end,1:2:end,i)-0.5).*(abs(GG3(1:2:end,1:2:end,i)-0.5))).^expo)
      xlabel('x (deg)'),ylabel('y (deg)')
      axis([0 2 0 2 -0.5.^expo 0.5.^expo]),axis square
      view([-75 15])
      subplot(235)
      GG3=g5;
      colormap([0 0 1;0 0 1])
      mesh(x(1,1:2:75),y(1:2:end,1),2*(0.25+sign(GG3(1:2:end,1:2:end,i)-0.5).*(abs(GG3(1:2:end,1:2:end,i)-0.5))).^expo)
      xlabel('x (deg)'),ylabel('y (deg)')
      axis([0 2 0 2 -0.5.^expo 0.5.^expo]),axis square
      view([-75 15])
      subplot(236)
      GG3=g6;
      colormap([0 0 1;0 0 1])
      mesh(x(1,1:2:75),y(1:2:end,1),2*(0.25+sign(GG3(1:2:end,1:2:end,i)-0.5).*(abs(GG3(1:2:end,1:2:end,i)-0.5))).^expo)
      xlabel('x (deg)'),ylabel('y (deg)')
      axis([0 2 0 2 -0.5.^expo 0.5.^expo]),axis square
      view([-75 15])
      set(8,'color',[1 1 1])
      lala = getframe(8);
      writeVideo(vidObj,lala);
end
close(vidObj);

figure(9), show_fft3(G1,fsx,fst,9),axis square,view(view_fourier),set(9,'color',[1 1 1])
figure(10),show_fft3(G2,fsx,fst,10),axis square,view(view_fourier),set(10,'color',[1 1 1])
figure(11),show_fft3(G3,fsx,fst,11),axis square,view(view_fourier),set(11,'color',[1 1 1])
figure(12),show_fft3(G4,fsx,fst,12),axis square,view(view_fourier),set(12,'color',[1 1 1])
figure(13),show_fft3(G5,fsx,fst,13),axis square,view(view_fourier),set(13,'color',[1 1 1])
figure(14),show_fft3(G6,fsx,fst,14),axis square,view(view_fourier),set(14,'color',[1 1 1]) 

 %% Apply MT to stimulus
 
 r1 = real(ifft3( G1.*FTY ,1));
 r2 = real(ifft3( G2.*FTY ,1));
 r3 = real(ifft3( G3.*FTY ,1));
 r4 = real(ifft3( G4.*FTY ,1));
 r5 = real(ifft3( G5.*FTY ,1));
 r6 = real(ifft3( G6.*FTY ,1));

Y1 = then2now(r1,75);
Y2 = then2now(r2,75);
Y3 = then2now(r3,75);
Y4 = then2now(r4,75);
Y5 = then2now(r5,75);
Y6 = then2now(r6,75);

m = min([Y1(:);Y2(:);Y3(:);Y4(:);Y5(:);Y6(:)]);
M = max([Y1(:);Y2(:);Y3(:);Y4(:);Y5(:);Y6(:)]);

Y1 = (Y1-m)/(M-m);
Y2 = (Y2-m)/(M-m);
Y3 = (Y3-m)/(M-m);
Y4 = (Y4-m)/(M-m);
Y5 = (Y5-m)/(M-m);
Y6 = (Y6-m)/(M-m);

C1 = cat(2,Yn,Y1,Y2,Y3);
C2 = cat(2,ones(75,75,64),Y4,Y5,Y6);
C = cat(1,C1,C2);

implay(C)

name = 'MT_movie3'; 
MOV = build_achrom_movie_avi(C,min(C(:)),max(C(:)),columns_x*3,15,fst,1,folder,name);

 