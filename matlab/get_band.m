function B = get_band( bands, b, o )

oc = min( o, bands.sz(b) );

B = pyrBand( bands.pyr, bands.pind, sum(bands.sz(1:(b-1)))+oc );

end