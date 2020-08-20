 function ea = average_deviation_wavelet(e,ind)

% 
% AVERAGE_DEVIATION_WAVELET: computes the amplitude per subband in steerable wavelet
%  
% Given a set of steerable pyramid vectors (columns on the matrix "e"), each 
% with a subband structure described by the matlabpyrtools parameter "ind", 
% the equivalent set of vectors with average amplitude per subband in each 
% coefficient is given by:
% 
%    e_a = average_deviation_wavelet(e,ind)
% 

n_sub = length(ind(:,1));
d = sum(prod(ind,2));
ea = 0*e;

for nband_f = 1:n_sub
    indi_f = pyrBandIndices(ind,nband_f);
    ea(indi_f,:) = repmat(mean(e(indi_f,:)),[length(indi_f) 1]);
end