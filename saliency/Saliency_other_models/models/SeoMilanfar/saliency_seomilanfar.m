
function [smap] = saliency_seomilanfar(input_image,image_path)

% parameters for local self-resemblance

param.P = 3; % LARK window size
param.alpha = 0.42; % LARK sensitivity parameter
param.h = 0.2; % smoothing parameter for LARK
param.L = 7; % # of LARK in the feature matrix 
param.N = 3; % size of a center + surrounding reagion for computing self-resemblance
param.sigma = 0.07; % fall-off parameter for self-resemblamnce **For visual purpose, use 0.2 instead of 0.07**

% parameters for global self-resemblance

param1.P = 3; % LARK window size
param1.alpha = 0.42; % LARK sensitivity parameter
param1.h = 0.2; % smoothing parameter for LARK
param1.L = 7; % # of LARK in the feature matrix 
param1.N = inf; % size of a center + surrounding region for computing self-resemblance
param1.sigma = 0.07; % fall-off parameter for self-resemblamnce. **For visual purpose, use 0.2 instead of 0.06**



% get saliency map:
smap = SaliencyMap(input_image,[64 64],param1);

end

