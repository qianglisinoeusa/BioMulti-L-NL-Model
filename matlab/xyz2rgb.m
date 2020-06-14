function RGB = xyz2rgb( XYZ )
% Transform image color space using matrix M
% dest = M * src

M = [ 3.240708 -1.537259 -0.498570;
    -0.969257  1.875995  0.041555;
    0.055636 -0.203996  1.057069 ];

RGB = reshape( (M * reshape( XYZ, [size(XYZ,1)*size(XYZ,2) 3] )')', ...
    [size(XYZ,1) size(XYZ,2) 3] );
end
