function [V, dV_x, dV_y]= cleComputePotential(sal, tau_V, rows, cols)
% cleComputePotential - Computes the random walk potential using a saliency map
%
% Synopsis
%          [V, dV_x, dV_y]= cleComputePotential(sal, tau_V, rows, cols)
%
% Description
%   Computes the random walk potential as the function 
%   $$ V(x,y)=\exp(-\tau _{V} s(x,y)) $$
%   of a saliency map $s$
%
% Inputs ([]s are optional)
%   (matrix ) sal        rows x cols salience map 
%   (double) tau_V       damping parameter
%   (int)    rows        number of rows
%   (int)    cols        number of columns
%
% Outputs ([]s are optional)
%   
%   (matrix ) V          rows x cols  Potential matrix
%   (matrix ) dV_x       rows x cols  matrix of the x component of the
%                        potential gradient grad(V)
%   (matrix ) dV_y       rows x cols  matrix of the y component of the
%                        potential gradient grad(V)
%
%
% See also
%   cleGenerateScanpath
%
% Requirements
%   Image Processing toolbox
%   Statistical toolbox

% References
%   [1] G. Boccignone and M. Ferraro, Modelling gaze shift as a constrained
%       random walk, Physica A, vol. 331, no. 1, pp. 207-218, 2004.
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
V = exp(-tau_V*sal)*100;

%precomputing potential gradients

% Construct diffl which is the same as V but
% has an extra padding of zeros around it.
diffl = zeros(rows+2, cols+2);
diffl(2:rows+1, 2:cols+1) = V;

% North, South, East and West differences
deltaN = diffl(1:rows,2:cols+1) - V;
deltaS = diffl(3:rows+2,2:cols+1) - V;
deltaE = diffl(2:rows+1,3:cols+2) - V;
deltaW = diffl(2:rows+1,1:cols) - V;

dV_x = (deltaW + deltaE)/2 ;
dV_y = (deltaS + deltaN)/2;

end
