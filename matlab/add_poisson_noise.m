function r_wn = add_poisson_noise(r_w,alpha_noise)

% ADD_POISSON_NOISE adds neural-like noise to a V1 representation computed
% with unit-norm receptive fields (see help of apply_V1 for details on the normalization)
% 
% Amplitude of Poisson noise (e.g. its standard deviation) depends on 
% the signal amplitude, |r|: the bigger the signal, the bigger the noise. 
% The magnitude of signal/noise ratio is controlled by the so-called Fano
% factor, F:
%
%             sigma_noise = sqrt( F * |r| )
%
% Therefore in Matlab:
%
%        r_n = r + sqrt(F*r).*randn(size(r));
%
% In our case, for a set of deterministic V1 responses in the vector r_w 
% (steerable wavelet format, see matlabpyrtools), we have:
%
%           r_wn = add_poisson_noise(r_w, F)
%
% where: 
%       r_w = deterministic V1 responses
%         F = Fano factor
%      r_wn = noisy V1 responses
%

r_wn = r_w + sqrt(alpha_noise*abs(r_w)).*randn(size(r_w));