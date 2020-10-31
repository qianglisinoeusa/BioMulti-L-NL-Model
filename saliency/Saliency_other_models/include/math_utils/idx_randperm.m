function [ permutation ] = idx_randperm(vector_length, r_length, notequal_idx )

    
    permutation = [];
    while length(permutation) < r_length
        rvalue = randi(vector_length);
        if rvalue ~= notequal_idx
               permutation = [permutation rvalue];
        end
        
    end

end

