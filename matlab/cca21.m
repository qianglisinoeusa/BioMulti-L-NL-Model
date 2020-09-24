% ----------------------------------------
%
% [s1_cca s2_cca ccaResults]= cca21(z1,z2)
%
%  z1, z2: whitened, zero mean data
% 
% ccaResults contains the SVD of the 
% cross-correlation matrix:
% K21 = U21 S21 V21^T, 
% or in the notation in the paper:
% Kad = Qa S Qd^T
%
% filter_A_cca = V21'*Wm1; % rows are the filters
% filter_D_cca = U21'*Wm2;

% feature_A_cca = U1*D1.^(0.5)*V21; % columns are the features
% feature_D_cca = U2*D2.^(0.5)*U21;
%  
% ----------------------------------------


function [s1_cca s2_cca ccaResults]= cca21(z1,z2)
    
    T = size(z1,2);
    K21 = z2*z1'/T;
    [U21 S21 V21] = svd(K21);
    
    % "aligned" coordinates
    s2_cca = U21'*z2;
    s1_cca = V21'*z1;
   
    ccaResults.U21 = U21;
    ccaResults.S21 = S21;
    ccaResults.V21 = V21;
    
  