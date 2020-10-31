function [  ] = fig2png( fig_path , out_path)
    if nargin < 2, out_path=[fig_path '.png']; end
    
    close all;
    try
        
        figs = openfig(fig_path);
        for K = 1 : length(figs)
           filename = out_path;
           disp(['Writing ' filename]);
           saveas(figs(K), filename);
        end
    catch
        fig2png( fig_path , out_path);
    end
end

