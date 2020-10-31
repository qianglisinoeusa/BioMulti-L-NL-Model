function [ cmap_union ] = cmap_union( cmaps_cell )
            cmap_union = [0 0];

            for l=1:length(cmaps_cell)
                cmap_union = union(cmap_union(:,:),cmaps_cell{l},'rows');

            end
            cmap_union(1,:) = [];
end

