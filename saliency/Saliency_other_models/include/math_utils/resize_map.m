
function [map1_resized] = resize_map(map1, map2)

            if isequal(size(map1(:,:,1)),size(map2(:,:,1))) == 0
                map1_resized = imresize(map1,[size(map2,1) size(map2,2)]);
                %disp(['resized map from:' int2str(size(map1,1)) 'x' int2str(size(map1,2)) ' to ' int2str(size(map1_resized,1)) 'x' int2str(size(map1_resized,2))  ]);
                %close all force;
                %imtool(map1);
                %imtool(map1_resized);
            else
                map1_resized = map1;
            end

end
