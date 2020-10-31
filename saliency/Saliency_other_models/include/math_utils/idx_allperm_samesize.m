function [ permutation ] = idx_allperm_samesize( vector_length,r_size, sizes, notequal_idx )
    
    
    permutation = [];
    for i=1:vector_length
        if i ~= notequal_idx && sizes(i,1) == r_size(1) && sizes(i,2) == r_size(2)
               permutation = [permutation i];
        end
        
    end
    

end

