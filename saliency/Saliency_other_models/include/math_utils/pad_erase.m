function [ output_image ] = pad_erase( input_image, limits )


output_image = input_image(limits(1):limits(2),limits(3):limits(4),:);

end

