function [MSCN_img]= MSCN(img)
    %Psychology named Divisive Normalization
    %Computer named Local mean and local normalization
    window = fspecial('gaussian',7,7/6); 
    window = window/sum(sum(window));
    mu = filter2(window, img, 'same');
    mu_sq = mu.*mu;
    sigma = sqrt(abs(filter2(window, ...
    img.*img, 'same') - mu_sq));
    imgMinusMu = (img-mu);
    MSCN_img =imgMinusMu./(sigma +1);

end