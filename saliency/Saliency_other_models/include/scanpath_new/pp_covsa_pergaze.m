function [ rho, pval, rhos] = pp_corrsa( scanpath1, scanpath2, ff_flags )
    if nargin<3, firstfixation_flag_default; end
    
    rho={};
    pval={};
    if iscell(scanpath1) && iscell(scanpath2)
        scanpath1_pergazes=scanpath_gazes(scanpath1);
        scanpath2_pergazes=scanpath_gazes(scanpath2);
        for g=1:min([length(scanpath1_pergazes) length(scanpath2_pergazes)])
            [rhos{g},pvals{g}]=covsa(scanpath1_pergazes{g},scanpath2_pergazes{g},ff_flags);
        end
        rho=nanmean(cell2mat(rhos));
        pval=nanmean(cell2mat(pvals));
        
    elseif iscell(scanpath1) && ~iscell(scanpath2)
        scanpath1_pergazes=scanpath_gazes(scanpath1);
        for g=1:length(scanpath1_pergazes)
            [rhos{g},pvals{g}]=corrsa(scanpath1_pergazes{g},scanpath2,ff_flags);
        end
        rho=nanmean(cell2mat(rhos));
        pval=nanmean(cell2mat(pvals));
        
    elseif ~iscell(scanpath1) && iscell(scanpath2)
        scanpath2_pergazes=scanpath_gazes(scanpath2);
        for g=1:length(scanpath2_pergazes)
            [rhos{g},pvals{g}]=covsa(scanpath1,scanpath2_pergazes{g},ff_flags);
        end
        rho=nanmean(cell2mat(rhos));
        pval=nanmean(cell2mat(pvals));
    else
        [rho,pval]=covsa(scanpath1,scanpath2,ff_flags);
        rhos=NaN;
    end

end

