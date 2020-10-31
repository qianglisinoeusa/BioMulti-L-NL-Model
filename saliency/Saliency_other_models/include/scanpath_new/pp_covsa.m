function [ rho, def,def] = pp_covsa( scanpath1, scanpath2, ff_flags )
    if nargin<3, firstfixation_flag_default; end
    
    def=NaN;
    rho=NaN;
    if iscell(scanpath1) && iscell(scanpath2)
        for p=1:length(scanpath1)
            [rho{p}]=covsa(scanpath1{p},scanpath2{p},ff_flags);
        end
        rho=nanmean(cell2mat(rho));
        
    elseif iscell(scanpath1) && ~iscell(scanpath2)
        for p=1:length(scanpath1)
            [rho{p}]=covsa(scanpath1{p},scanpath2,ff_flags);
        end
        rho=nanmean(cell2mat(rho));
        
    elseif ~iscell(scanpath1) && iscell(scanpath2)
        for p=1:length(scanpath2)
            [rho{p}]=covsa(scanpath1,scanpath2{p},ff_flags);
        end
        rho=nanmean(cell2mat(rho));
    else
        [rho]=covsa(scanpath1,scanpath2,ff_flags);
    end
    
end

