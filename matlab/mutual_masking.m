function m = mutual_masking( b, o )
        m = min( abs(get_band(bands.sz,b,o)) );
        % simplistic phase-uncertainty mechanism 
        % TODO - improve it
        
        if( metric_par.do_si_gauss ) 
            m = blur_gaussian( m, 10^metric_par.si_size );
        
        else
            F = ones( 3, 3 );
            m = conv2( m, F/numel(F), 'same');
        end
       

end