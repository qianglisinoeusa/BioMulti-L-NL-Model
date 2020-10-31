function [foaStore, sal] = cleGenerateScanpath(config_file, nOBS,nFixations)
% cleGenerateScanpath - Generates a scanpath computing eye movements as 
%                       Levy flight on a saliency map
%
% Synopsis
%          cleGenerateScanpath(config_file, [nOBS])
%
% Description
%   Generates a visual scanpath by computing gaze shifts as Levy flights on
%   any kind of saliency map (bottom-up or top-down) computed for the 
%   given image. Basically a simple, but slightly enhanced, implementation of the algorithm
%   described in the original paper of  Boccignone & Ferraro [1].
%   The only variant with respect to [1] is the use of an internal
%   simulation step along which a number of candidate gaze shifts is
%   sampled [2].
%
%   See the comments in each routine for details of what it does
%   Settings for the experiment should be held in the configuration
%   file.
%
% Inputs ([]s are optional)
%   (string) config_file  the name of a configuration file in the .\config
%                         directory to be evaluated for setting the
%                         experiment parameters
%   (integer)[nOBS]       the number of virtual observers / generated scanpaths
%
% Outputs ([]s are optional)
%   
%   ....
%
% Example:
%   
%   cleGenerateScanpath('config',5);
%
% See also
%   runExperimentCLE
%   cleComputeSaliency
%   cleComputePotential
%   cleShiftGazeLevy
%   cleWeightSal
%
% Requirements
%   Image Processing toolbox
%   Statistical toolbox

% References
%   [1] G. Boccignone and M. Ferraro, Modelling gaze shift as a constrained
%       random walk, Physica A, vol. 331, no. 1, pp. 207-218, 2004.
%   [2] G. Boccignone and M. Ferraro, Feed and fly control of visual 
%       scanpaths for foveation image processing, Annals of telecommunications - 
%       Annales des t�l�communications 
%       2012 (in press).
%   
%
% Authors
%   Giuseppe Boccignone <Giuseppe.Boccignone(at)unimi.it>
%
% License
%   The program is free for non-commercial academic use. Please 
%   contact the authors if you are interested in using the software
%   for commercial purposes. The software must not modified or
%   re-distributed without prior permission of the authors.
%
% Changes
%   20/01/2011  First Edition
%

addpath(genpath('config'));
addpath(genpath('saltool'));
addpath(genpath('stats'));
addpath(genpath('visualization'));

%error(nargchk(1, 2, nargin));
if ~exist('nOBS', 'var') || isempty(nOBS)
    nOBS = 1;
end
if ~exist('nFixations', 'var') || isempty(nOBS)
    nFixations = 10;
end

% Evaluating the configuration script containing all the parameters of the
% experiment
try
    eval(config_file);
catch ME
    disp('error in evaluating config script')
    ME
end

numStep=nFixations;

fprintf('\n Generating scanpath for  model observer %d ---------\n', nOBS);

%----------------------------------------------------------------------
% Loading the image 
%----------------------------------------------------------------------
fprintf('\n Loading the image... \n');
originalImage = imread(figname);
[rows cols C]   = size(originalImage);
if C<3, 
    originalImage(:,:,2)=originalImage(:,:,1); 
    originalImage(:,:,3)=originalImage(:,:,1);
    C=3; 
end

if VISUALIZE_RESULTS_ONLINE
    %----------------------------------------------------------------------
    % The art corner: showing the original image
    %----------------------------------------------------------------------
    
    % Displaying relevant figures.
    scrsz = get(0,'ScreenSize');
    figure('Position',[1 scrsz(4) scrsz(3) scrsz(4)])
    hold on

    % Here the original frame.
    subplot(NUMLINE, NUMPICSLINE, ORIG_POS); 
    sc(originalImage);
    label(originalImage, 'Original image');
end


% Determining the foaSize as in the original  Itti & Koch's PAMI paper
foaSize = 1/6 * min(rows, cols);

if SAVE_FOA_ONFILE
    foaStore=[];
end


%----------------------------------------------------------------------
% Build the saliency map $$s(�)$$ of the image
%----------------------------------------------------------------------

fprintf('\n Build the saliency map... \n');
% for timing...
sum_t_sal_elapsed=0;
sum_t_gaze_elapsed=0;
sum_t_total_elapsed=0;
t_sal_elapsed=0;

if ~precomputedSalMap    
    t_sal_start = tic;
    
    % Calling the appropriate salience computation method
	sal= cleComputeSaliency(originalImage,SALIENCY_TYPE);
    
    t_sal_elapsed = toc(t_sal_start);  
    fprintf('\n-------Time to make saliency: %f \n',t_sal_elapsed);
                
else
    % if already computed, load the salience previously stored as an image
    saliencyMap=imread(salname);
    sal = im2double(imresize(saliencyMap, [rows cols], 'bilinear'));
end

sum_t_sal_elapsed=sum_t_sal_elapsed+t_sal_elapsed;

% Normalize the salience in [0,100] so that different methods will provide
% a map in the same range
maxsal=max(max(sal));
minsal=min(min(sal));
sal= 100 * (sal-minsal)./(maxsal-minsal);

if VISUALIZE_RESULTS_ONLINE
show_sMap=sal;
tempIm=cat(3,show_sMap, originalImage); 
sc(tempIm,'prob_jet');  
label(tempIm, 'Saliency map');
end

if VISUALIZE_RESULTS_ONLINE
    %----------------------------------------------------------------------
    % The art corner: showing the saliency
    %----------------------------------------------------------------------
    
    subplot(NUMLINE, NUMPICSLINE, SAL_POS) ;
    
    
end

if (SAVE_SAL_IMG) && (nOBS==1) %just save the first time, salmap is the same for all observers
        [X,MAP]= frame2im(getframe);
        FILENAME=[RESULT_DIR 'SAL' IMAGE_NAME '.tif'];
        imwrite(X,FILENAME,'tif');
end

%----------------------------------------------------------------------
% Compute potential V according to Eq. (2):
%     $$ V(x,y)=\exp(-\tau _{V} s(x,y)) $$
%     V is modelled as a decreasing function of the saliency field s
%----------------------------------------------------------------------
fprintf('\n Compute potential V... \n');
[V, dV_x, dV_y]= cleComputePotential(sal, tau_V, rows, cols);


if VISUALIZE_RESULTS_ONLINE
    %----------------------------------------------------------------------
    % The art corner: plotting the potential landscape in 3D
    %----------------------------------------------------------------------
    subplot(NUMLINE, NUMPICSLINE, POT_POS) ;
    surf(double(V(1:4:rows,1:4:cols)),'FaceColor','interp',...
     'EdgeColor','none',...
     'FaceLighting','phong'),zlim([0 100]);
    axis tight
    view(-11,66)
    camlight left
    set(gca,'ydir','reverse');
    title('Potential landscape')
    pause(0.05)
    
end

if (SAVE_POT_IMG) && (nOBS==1) %just save the first time, potential is the same for all observers
[X,MAP]= frame2im(getframe);
FILENAME=[RESULT_DIR 'POT' IMAGE_NAME '.tif'];
imwrite(X,FILENAME,'tif');
end
%axis([-3 3 -3 3 -10 5])




%----------------------------------------------------------------------
% Set first FOA on image center
%----------------------------------------------------------------------
xc = round(rows/2);
yc = round(cols/2);


if VISUALIZE_RESULTS_ONLINE
    
    subplot(NUMLINE, NUMPICSLINE, SCAN_POS) ;
    sc(originalImage)
    hold on;   
    plot(yc,xc,'y.','MarkerSize',15);
    
    drawnow 
    
end

%----------------------------------------------------------------------
% Starting the Levy saccade generation loop for producing gaze-shift
%----------------------------------------------------------------------
fprintf('\n Starting to shift gaze... \n');   
x_old= xc; y_old= yc;
x_new= xc; y_new= yc;
foaCord=[x_new;y_new];

if SAVE_FOA_ONFILE
    foaStore=[foaStore foaCord];
end


oldDir=0;


% Tuning Levy flight parameters 
% to rescale r on the dimension of the FOA;
rescaleLenght= 2 * foaSize;
gamma_stable= (rescaleLenght)^2;     

% number of candidate gaze shifts  generated during internal simulation (see [2])        
n = NUM_SAMPLE_LEVY; 

t_gaze_start = tic;

% Gazing.........
for i=1:numStep
    if VERBOSE_CLE
      disp('RANDOM SEARCH: STARTING....');
    end
    
    %----------------------------------------------------------------------
    % Shifting the Gaze Levy via jump lenght
    %----------------------------------------------------------------------
    while (x_new==x_old && y_new==y_old)
        [x_new, y_new, candx, candy ] = cleShiftGazeLevy(alpha_stable, beta_stable, gamma_stable, delta_stable, n,...
                                               x_old, y_old, oldDir, k_P, k_R, dV_x, dV_y, h,...
                                               rows, cols, sal); 




        %----------------------------------------------------------------------
        % Starts the acceptance process:
        %     the flight is accepted according to a probabilistic rule 
        %     that depends on the gain of saliency and 
        %     on a temperature T , whose values determine the amount 
        %     of randomness in the acceptance  process. 
        %----------------------------------------------------------------------

        % Saliency gain evaluated within a Gaussian window centered on
        % the old and the new FOA
        sigma     = foaSize;  %scales the window to compute the average saliency
        w_sal_new = cleWeightSal(x_new,y_new,sal,sigma);
        w_sal_old = cleWeightSal(x_old,y_old,sal,sigma);
        deltaS    = w_sal_new - w_sal_old; %saliency gain   

        if( deltaS > 0) 
            %----------------------------------------------------------------------
            % straight gain of saliency: ACCEPTED! 
            %----------------------------------------------------------------------
            if VERBOSE_CLE
            disp('RANDOM SEARCH: ACCEPTED IMMEDIATELY!!');
            end
        else
            %----------------------------------------------------------------------
            % Metropolis step
            %----------------------------------------------------------------------       

            p  = exp(deltaS/T) ; 
            tr = min( p,1);

            rho = rand(1,1);
            if (rho < tr)

                if VERBOSE_CLE
                disp('RANDOM SEARCH: ACCEPTED BY METROPOLIS!!');
                end
            else
                if VERBOSE_CLE
                disp('RANDOM SEARCH: REJECTED BY METROPOLIS!! KEEPING OLD FOA');
                end
                x_new = x_old ;
                y_new = y_old;

            end
        end
    end
    if VISUALIZE_RESULTS_ONLINE
        subplot(NUMLINE, NUMPICSLINE, SCAN_POS) ;
        label(originalImage, 'CLE Scanpath');
        plot(y_new,x_new,'y.','MarkerSize',15);
        plot([y_old,y_new],[x_old,x_new],'y-','MarkerSize',1);
        drawnow

        hold on
    end
    % Computing the direction of flight
    isChangedPoint = (x_old ~= x_new) | (y_old ~= y_new);
    if isChangedPoint
         xx     = sqrt((x_old - x_new)^2); 
         yy     = sqrt((y_old - y_new)^2);
         newDir = atan(yy/xx);
    else
         newDir = oldDir;
    end
    oldDir = newDir; 

    % Storage of current FOA for the next step
    x_old= x_new;
    y_old= y_new;
    foaCord=[x_new;y_new];
    if SAVE_FOA_ONFILE
    foaStore=[foaStore foaCord];
    end
               
end % ----gaze shifting

% Visualizing elapsed times
fprintf('\n-------Time to make saliency: %f \n',t_sal_elapsed);
t_gaze_elapsed      = toc(t_gaze_start);  
fprintf('\n-------Time to gaze: %f \n',t_gaze_elapsed);
sum_t_gaze_elapsed  = sum_t_gaze_elapsed+t_gaze_elapsed;
total_elapsed       = t_gaze_elapsed  + t_sal_elapsed;
fprintf('\n-------Time total: %f \n',total_elapsed);
sum_t_total_elapsed = sum_t_total_elapsed+total_elapsed;

%----------------------------------------------------------------------
% RESULT STORAGE IF REQUESTED IN THE CONFIG SCRIPT
%----------------------------------------------------------------------
if VISUALIZE_RESULTS_ONLINE
        
        subplot(NUMLINE, NUMPICSLINE, SCAN_POS) ;       
        plot(y_new,x_new,'y.','MarkerSize',15);      
        drawnow
        hold on
end

if SAVE_SCANPATH_IMG
    
        [X,MAP]= frame2im(getframe);        
        SCANFILENAME=[RESULT_DIR  'CLESCAN_' EXPERIMENT_TYPE  '_RUN_' num2str(nOBS) '_NFOA_' num2str(numStep) '_' IMAGE_NAME '.tif'];        
        imwrite(X,SCANFILENAME,'tif');
end


if SAVE_FOA_ONFILE
    FOAFILENAME=[RESULT_DIR  'CLESCAN_' EXPERIMENT_TYPE '_RUN_' num2str(nOBS) '_NFOA_' num2str(numStep) '_' IMAGE_NAME '.txt'];
    save(FOAFILENAME,'foaStore','-ASCII')
end



fprintf('\n Model observer %d completed!---------\n', nOBS);


end %MAIN

    
