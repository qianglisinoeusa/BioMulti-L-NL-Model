function [ rvalue ] = drandperm(type, min, max, r_length, notequal_value )

    
    switch type
        case 'int'
            rvalue = round(rvalue);
        otherwise
    end

    rvalue = randperm(max,r_length);    
        
    if ismember(rvalue,notequal_value) == 1
        rvalue = drandperm(type, min, max,r_length, notequal_value );
    end

end

