function [ vert_1deg ] = vertangle( vertres,vertmeters,distmeters )

    vert_1deg = (vertres/vertmeters)*distmeters*tand(1);

end

