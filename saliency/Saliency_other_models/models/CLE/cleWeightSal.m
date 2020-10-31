function wsal = cleWeightSal(x,y,sal,sigma)
% cleWeightSal - weights the salience at a certain point
%
% Synopsis
%          wsal = cleWeightSal(x,y,sal,sigma)
%
% Description
%   weights the salience at a certain point (x,y) by using a Gaussian
%   windows centered at that point
%

%
% Inputs ([]s are optional)
%   (integer) (x,y)       the coordinates of the point
%                         experiment parameters
%   (matrix) sal          the salience map
%   (float)  sigma        the dimension of the window
%
% Outputs ([]s are optional)   
%   (float)  wsal         the weighted saliency
%
%
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

%win=round(sqrt(sigma));
win=round(sigma/2);
xwin = (x-win:1:x+win); 
ywin = (y-win:1:y+win)';

[rows cols]=size(sal);
if(isempty(find(xwin<1)) && isempty(find(xwin>rows)) && isempty(find(ywin<1)) && isempty(find(ywin>cols)))
    [X,Y] = meshgrid(xwin, ywin); 
    F = sal(xwin, ywin) .* exp(-(((X-x).^2 + (Y-y).^2)));
    wsal=sum(sum(F));
    return;
else
    wsal=sal(x,y);
end
return