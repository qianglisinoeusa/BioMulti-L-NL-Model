function RGB = display_model_srgb( sRGB )
a = 0.055;
thr = 0.04045;

RGB = zeros(size(sRGB));
RGB(sRGB<=thr) = sRGB(sRGB<=thr)/12.92;
RGB(sRGB>thr) = ((sRGB(sRGB>thr)+a)/(1+a)).^2.4;

RGB = 99*RGB + 1;

end