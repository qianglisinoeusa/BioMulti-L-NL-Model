function check_if_values_plausible( img, metric_par )
% Check if the image is in plausible range and report a warning if not.
% This is because the metric is often misused and used for with
% non-absolute luminace data.

if( ~metric_par.disable_lowvals_warning )
    if( max(img(:)) <= 1 ) 
        warning( 'hdrvdp:lowvals', [ 'The images contain very low physical values, below 1 cd/m^2. ' ...
            'The passed values are most probably not scaled in absolute units, as requied for this color encoding. ' ...
            'See ''doc hdrvdp'' for details. To disable this wanrning message, add option { ''disable_lowvals_warning'', ''true'' }.' ] );
    end
end

end
