function [ permutation ] = idx_allperm( vector_length,notequal_idx )
    
    
    permutation = [];
    for i=1:vector_length
        if i ~= notequal_idx
               permutation = [permutation i];
        end
        
    end
    

end

