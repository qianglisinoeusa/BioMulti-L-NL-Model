function [ mean_distance, std_distance, distances ] = pp_slanding_11( scanpath1, scanpath2, ff_flags )
    if nargin<3, ff_flags=[1 0]; end
    [ mean_distance, std_distance, distances ]=pp_slanding( scanpath1, scanpath2, ff_flags );
end

