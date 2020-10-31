function [ permutation ] = idx_randperm_samesize(vector_length, r_length, r_size, sizes, notequal_idx )

    
    permutation = [];
    while length(permutation) < r_length
        rvalue = randi(vector_length);
        if rvalue ~= notequal_idx && sizes(rvalue,1) == r_size(1) && sizes(rvalue,2) == r_size(2)
               permutation = [permutation rvalue];
        end
        
    end

end

