clear all; close all; clc;
% runExperimentCLE-  Top level script that runs a 
%                    Constrained Levy Exploration (CLE) experiment.
%                    The experiment consists in putting into action a
%                    defined number of 
%                    artificial observers, each generating a visual scanpath
%                    (a sequence of fixations and saccades) on a given
%                    image using a slightly enhanced version of the CLE method 
%                    described in Boccignone & Ferraro [1].
%                    Enhancements concern;
%                    - 1) Possibility of using more general alpha-stable
%                       distributions [2] rather then stick to the Cauchy
%                       distribution as in [1]
%                    - 2) An informed strategy is employed to sample the
%                       next gaze shift in that the choice of the next gaze location is
%                       obtained through an internal simulation step: a number n of candidates gaze
%                       shifts is preliminarly sampled and evaluated against a gain function [2]. 
%                       The best among n candidate shift is eventually retained 
%                    All paremeters defining the experiment are
%                    defined in the config_<type of experiment>.m script
%                    file
%
% See also
%   cleGenerateScanpath
%   config_<type of experiment>
%
% Requirements
%   Image Processing toolbox
%   Statistical toolbox

% References
%   [1] G. Boccignone and M. Ferraro, Modelling gaze shift as a constrained
%       random walk, Physica A, vol. 331, no. 1, pp. 207-218, 2004.
%   [2] G. Boccignone and M. Ferraro, Feed and fly control of visual 
%       scanpaths for foveation image processing, Annals of telecommunications - 
%       Annales des télécommunications 
%       2012 (in press).
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
%--------------------------------------------------------------------------
% Set here the total number of observers / scanpaths to be simulated
%--------------------------------------------------------------------------
totObservers = 5;

% Set the configuration filename  (parameters) of the experiment
configFileName = 'config_simple';

for nObs=1:totObservers
    % Generate and visualize a CLE scanpath
    %   Calling the overall routine cleGenerateScanpath that does everything, with 
    %   a configuration file: the routine will run each subsection of the gaze shift 
    %   scheme in turn.
    cleGenerateScanpath(configFileName, nObs);
    
end
