function [ centX, centY ] = calc_centroid( map , method)
    if nargin < 2, method='hist'; end
    

    switch method
        case 'hist'
            X_hist=sum(map,1);
            Y_hist=sum(map,2);
            X=1:size(map,2); Y=1:size(map,1);
            centX=round(sum(X.*X_hist)/sum(X_hist));
            centY=round(sum(Y'.*Y_hist)/sum(Y_hist));
        otherwise
            [max_num,ind]=max(map(:));
            [centX, centY]=ind2sub(size(map),ind);
    end

end

