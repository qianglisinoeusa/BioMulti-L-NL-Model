function [ rvalue ] = drand(type, min, max, notequal_value )

    rvalue = min + (max-min).*rand(1);
    
    switch type
        case 'int'
            rvalue = round(rvalue);
        otherwise
    end
    
        
    if ismember(rvalue,notequal_value) == 1
        rvalue = drand(type, min, max, notequal_value );
    end

end

