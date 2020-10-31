function [mix_smap] = mix_smaps(smaps,mode,part)
	if nargin < 2, mode='mean'; end
	if nargin < 3, part=size(smaps,3); end

    switch(mode)
            case 'max'
                mix_smap = get_smaps_max(smaps,part);
            case 'sum'
                mix_smap = get_smaps_sum(smaps,part);
            otherwise
                mix_smap = get_smaps_mean(smaps,part);
    end
end

