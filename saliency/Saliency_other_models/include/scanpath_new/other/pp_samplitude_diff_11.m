function [ mean_amplitude_diff, std_amplitude_diff, amplitudes_diff  ] = pp_samplitude_diff_01( scanpath1,scanpath2, ff_flags )
    if nargin<3, ff_flags=[1 1]; end
    [ mean_amplitude_diff, std_amplitude_diff, amplitudes_diff  ] =pp_samplitude_diff( scanpath1,scanpath2, ff_flags );
    

end

