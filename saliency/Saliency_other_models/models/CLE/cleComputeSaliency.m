function Sal = cleComputeSaliency(inImage,salType)
% cleComputeSaliency - a wrapper for salience computation
%
% Synopsis
%   Sal = cleComputeSaliency(inImage,salType)
%
% Description
%   The function is a simple wrapper for salience computation. Executes some kind
%   of salience computation algorithm which is defined from the parameter
%   salType by calling the appropriate function. Here for simplicity only
%   the Spectrual Residual method has been considered. 
%   If other methods need to be experimented, then you should extend the if...elseif...end
%   control structure
%
% Inputs ([]s are optional)
%   (matrix) inImage      the original image
%   (string) salType      the chosen method
%
% Outputs ([]s are optional)
%   (matrix) Sal          the saliaency map
%   ....
%
% Examples
%   sal= cleComputeSaliency(originalImage,'SPECTRAL_RES');
%
% See also
%   SpectralResidualSaliency
%
% Requirements
%   SpectralResidualSaliency (./saltool/SpectralR/)

% References
%   <Reference List like Papers>
%   ...
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
if strcmp(salType,'SPECTRAL_RES')
      % using the Spectral Residual method
      Sal = SpectralResidualSaliency(inImage);  
             
else  
      %----------------------------------------------------------------------
      % Add any suitable method to build 
      % the saliency map $$s(·)$$ of the image
      %----------------------------------------------------------------------
      disp('UNKNOWN SALIENCY METHOD')
end
