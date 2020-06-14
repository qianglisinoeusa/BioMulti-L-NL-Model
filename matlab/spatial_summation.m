function Y = spatial_summation( X, sigma )
% Essentilally a non-normalized Gaussian filter
% 

ksize = round(sigma*6);
h = fspecial( 'gaussian', ksize, sigma );
h = h / max(h(:));
Y = imfilter( X, h, 'replicate' );
    
end
