function [ rho, pval, def] = pp_corrsa( scanpath1, scanpath2, ff_flags )
    if nargin<3, firstfixation_flag_default; end
    
    rho={};
    pval={};
    def=NaN;
    if iscell(scanpath1) && iscell(scanpath2)
        for p=1:length(scanpath1)
            [rho{p},pval{p}]=corrsa(scanpath1{p},scanpath2{p},ff_flags);
        end
        rho=nanmean(cell2mat(rho));
        pval=nanmean(cell2mat(pval));
        
    elseif iscell(scanpath1) && ~iscell(scanpath2)
        for p=1:length(scanpath1)
            [rho{p},pval{p}]=corrsa(scanpath1{p},scanpath2,ff_flags);
        end
        rho=nanmean(cell2mat(rho));
        pval=nanmean(cell2mat(pval));
        
    elseif ~iscell(scanpath1) && iscell(scanpath2)
        for p=1:length(scanpath2)
            [rho{p},pval{p}]=corrsa(scanpath1,scanpath2{p},ff_flags);
        end
        
        rho=nanmean(cell2mat(rho));
        pval=nanmean(cell2mat(pval));
    else
        [rho,pval]=corrsa(scanpath1,scanpath2,ff_flags);
    end

end

