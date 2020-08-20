%%
%%  STATISTICS OF FOSTER DATA IN A PSYCHOPHYSICALLY TUNED WILSON-COWAN CASCADE
%%
%%    In this script I gather the data and compute the relevant statistics
%%
%%      Input data: 
%%         * Original images and divisive normalization cascade:
%%              /media/disk/vista/Papers/2017_Information_Flow/DataFoster/processed_40x40/samples_general_40x40_X.mat
%%                                                                                        samples_general_40x40_X_y1.mat

% Luminace images and responses (from y1 to x4)

%% 1.Load data batch-wise and put it in a single vector
timeFiles = '/media/disk/databases/BBDD_video_image/Image_Statistic/DataFoster/processed_40x40/';% Valencia

name = 'samples_general_40x40_';

y1_original = [];
y1 = [];
x1 = [];
x3 = [];
y4 = [];
x4 = [];
x4wc = [];

inicio = 0;
for batch=1:20 % Loop over batches
    i
    if(batch~=3)
        % Load image batch here
        load([timeFiles,name,num2str(batch),'.mat'])
        %load([timeFiles,nameJ,num2str(batch),'.mat'])
        display(['Batch ',num2str(batch),' loaded'])    
        numImages = size(samplesA4x,2);

        x1 = [x1 samplesA1x/260^1.25];   % See notebook (20 jun 2019) for the mistake in normalization of brightness
        x3 = [x3 samplesA3x];
        x4 = [x4 samplesA4x];
        y4 = [y4 samplesA4y];

        % Load image batch here
        load([timeFiles,name,num2str(batch),'_y1.mat'])

        y1_original = [y1_original samplesA1y_sin_v];
        y1 = [y1 samplesA1y];
        
        load(['/media/disk/vista/Papers/2019_Information_Flow_Wilson_Cowan/Resuts_integration_TF_2/',num2str(batch)])
        x4wc = [x4wc integrated];
        
        cuals(batch).indices = inicio+1:inicio+length(samplesA1y(1,:));
        inicio = inicio+length(samplesA1y(1,:));
    else
        cuals(batch).indices = [];
    end
end

ind =[40    40;
    40    40;
    40    40;
    40    40;
    40    40;
    20    20;
    20    20;
    20    20;
    20    20;
    10    10;
    10    10;
    10    10;
    10    10;
     5     5];
 
%%%% MAL: 2415 - 2484 

y1_original_ok = y1_original(:,[1:2415 2485:5244]);
y1_ok = y1(:,[1:2415 2485:5244]);
x1_ok = x1(:,[1:2415 2485:5244]);
x3_ok = x3(:,[1:2415 2485:5244]);
y4_ok = y4(:,[1:2415 2485:5244]);
x4_ok = x4(:,[1:2415 2485:5244]);
x4wc_ok = sign(x4_ok).*x4wc(:,[1:2415 2485:5244]);

samples_foster_y1 = [];
for i=1:length(y1_ok(1,:))
    y1 = reshape(y1_ok(:,i),[40 40]);
    y1 = im2col(y1,[3 1]);
    samples_foster_y1 = [samples_foster_y1 y1];
end

[n,x]=hist(samples_foster_y1(:),1000);
cdf = made_monotonic(cumsum(n)/sum(n));
figure,plot(x,cdf);
ymax = interp1(cdf,x,0.975)

samples_foster_y1 = samples_foster_y1/ymax;

save(['/media/disk/vista/Papers/2019_Information_Flow_Wilson_Cowan/3D_example_jesus/data_foster_3d'],'samples_foster_y1')

