function [ rho, pval] = covsa( scanpath1,scanpath2, ff_flags)
if nargin < 3, firstfixation_flag_default; end
    if ff_flags(1) == 1 && size(scanpath1,1)>0
        scanpath1(1,:)=[];
    end
    if ff_flags(2) == 1 && size(scanpath2,1)>0
        scanpath2(1,:)=[];
    end

min_gazes = min(size(scanpath1,1),size(scanpath2,1));
[~,~,amplitudes1]=samplitude(scanpath1,ff_flags(1));
[~,~,amplitudes2]=samplitude(scanpath2,ff_flags(2));

if length(amplitudes1) ~= length(amplitudes2)
    min_gazes_amp=min(length(amplitudes1),length(amplitudes2));
    amplitudes1(min_gazes_amp+1:end)=[];
    amplitudes2(min_gazes_amp+1:end)=[];
end

rho=NaN;
pval=NaN;
if length(amplitudes1)>0 && length(amplitudes2)>0
    [rho,pval]=corr(amplitudes1',amplitudes2');
end



end

