
function sz = get_band_size( bands, b, o )
sz = bands.pind(sum(bands.sz(1:(b-1)))+o,:);
end