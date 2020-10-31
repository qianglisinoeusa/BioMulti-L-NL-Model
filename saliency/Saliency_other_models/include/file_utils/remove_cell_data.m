function [cell_out] = remove_cell_data( cell_in, index )

    matrix = char(cell_in');
    matrix(index,:) = [];
    cell_out = cellstr(matrix);
end

