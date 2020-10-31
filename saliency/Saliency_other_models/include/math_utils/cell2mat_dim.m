function [ mat_out ] = cell2mat_dim( cell_in )
    
    M = size(cell_in{1},1);
    N = size(cell_in{1},2);
    D = length(cell_in);
    
    mat_out = zeros(M,N,D);
    for i=1:length(cell_in)
        mat_out(:,:,i) = cell_in{i};
    end

end

