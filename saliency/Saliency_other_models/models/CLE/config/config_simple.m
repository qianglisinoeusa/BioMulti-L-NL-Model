% Global configuration file 

% Single most important setting - the overall experiment type
% used by runExperimentCLE.m
% Holds all settings used in all parts of the code, enabling the exact
% reproduction of the experiment at some future date.

% COPY AND SUITABLY CHANGE IT TO PERFORM ANOTHER EXPERIMENT...

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




%----------------------------------------------------------------------
% EXPERIMENT PERFORMED 
%----------------------------------------------------------------------

EXPERIMENT_TYPE = 'SIMPLE'; % naming the experiment we are performing

% Are we using a precomputed salience map?
precomputedSalMap = false;   % if true uses a precomputed Salience Map 
							% stored in the form of an image
% Do we need some verbosity?
VERBOSE_CLE = 0;

%----------------------------------------------------------------------
% DIRECTORIES 
%----------------------------------------------------------------------

% Directory holding the code
RUN_DIR = [pwd '/']; %RUN_DIR will be the current directory

% Directory holding all the source images
IMAGE_DIR= [RUN_DIR 'img/'];

% Directory holding all the experiment results
RESULT_DIR = [RUN_DIR 'results/'];

% Directory holding code for generating levy flights
LEVY_DIR = [RUN_DIR 'stats/'];


%----------------------------------------------------------------------
% DATA TO PROCESS - please change at your convenience
%----------------------------------------------------------------------

IMAGE_NAME = 'default';          %original image name

% Making the path to the image
figname = [IMAGE_DIR IMAGE_NAME '.png'];

 


%----------------------------------------------------------------------
% GENERAL PARAMETERS OF THE ALGORITHM
%----------------------------------------------------------------------

%global foaSize

%--- Salience map definition
if precomputedSalMap
	SALMAP_NAME   = [IMAGE_NAME 'salmap.jpg'];  % pre-computed saliency map name
    % making the path to the map
    salname       = [IMAGE_DIR SALMAP_NAME];    %complete pathname
else
    % which method to compute the saliency?
	SALIENCY_TYPE = 'SPECTRAL_RES';          

end

%---  Damping parameter in potential 
%  $$ V(x,y)=\exp(-\tau _{V} s(x,y))$$
tau_V= 0.01; %original 0.01

%--- Random walk parameters

firstFOAonCENTER = true; %if true sets the first Foa on frame center

mix=0.5;
k_P =   mix;  %weighting the potential contribution to dynamics
k_R = 1-mix;  %weighting the Levy random component contribution to dynamics

beta_P=1.0;

numStep = 10; %number of gaze shifts

NUM_SAMPLE_LEVY = 50; %number of possible gaze shifts  generated

h = 0.1; % time step for the Euler discretization

% Setting the parameters of the alpha-stable random component

% alpha is the exponent (alpha=2 for gaussian, alpha=1 for cauchian)
alpha_stable = 1.0;
% gamma_stable is related to the standard deviation
gamma_stable = 1;
% beta is the  symmetry: 0 provides a symmetric alpha-stable distribution
beta_stable  = 0;   
% delta is the location parameter (for no drift, set to 0)
delta_stable = 0; 

%temperature for metropolis
% T=1; 
% T=10; 
% T=20; 
T=25; 
%T=100;     


%----------------------------------------------------------------------
% FOR VISUALIZATION AND RESULT STORAGE
%----------------------------------------------------------------------
VISUALIZE_RESULTS_ONLINE = true; %if true visualizing on line gaze shifts

if VISUALIZE_RESULTS_ONLINE
    font_size=18; line_width=2 ; marker_size=16;
    % subfigure positioning 
    NUMLINE=2;
    NUMPICSLINE=2;
    ORIG_POS = 1;
    SAL_POS  = 2;
    POT_POS  = 3;
    SCAN_POS = 4;


    
end
%if true saving pictures of saliency, potential, and scanpath  
SAVE_SAL_IMG      = false;
SAVE_POT_IMG      = false;
SAVE_SCANPATH_IMG = false;

SAVE_FOA_ONFILE   = true; % if true saving the (x,y) coordinates of the gaze positions
