function bands = set_band( bands, b, o, B )

bands.pyr(pyrBandIndices(bands.pind,sum(bands.sz(1:(b-1)))+o)) = B;

end
