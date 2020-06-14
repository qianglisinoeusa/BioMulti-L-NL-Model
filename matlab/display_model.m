function L = display_model( V, gamma, peak, black_level )
L = peak * V.^gamma + black_level;
end