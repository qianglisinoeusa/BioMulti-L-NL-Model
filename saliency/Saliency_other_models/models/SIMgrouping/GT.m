function [w c] = GT(image, wlev, nu_0, mode, window_sizes)
% Apply wavelet and grouplet decompositions to image, weight grouplet
% coefficients using ECSF, reconstruct wavelet plane using interpolation
%
% outputs:
%   w: cell array of length wlev, containing wavelet planes in 3
%   orientations.
%   c: cell array of length c, containing residual planes.
%
% inputs:
%   image: input image to be decomposed.
%   wlev: # of wavelet levels.
%   nu_0: offset of peak spatial frequency for CSF in case of image
%   resizing
%   mode: type of channel i.e. colour or intensity
%   window sizes: window sizes for computing relative contrast; suggested 
%   value of [17 37]

% pad image so that dimensions are powers of 2:
image = add_padding(image);

% Defined 1D Gabor-like filter:
h = [1./16.,1./4.,3./8.,1./4.,1./16.];

energy = sum(h);
inv_energy = 1/energy;
h = h*inv_energy;
w = cell(wlev,1);
c = cell(wlev,1);

for s = 1:wlev
    [m, ~]  = size(image);
    inv_sum = 1/sum(h);
   
    HF   = symmetric_filtering(image, h)*inv_sum;            % blur horizontally
    GF   = symmetric_filtering(image, h')*inv_sum;           % blur vertically
    allF = symmetric_filtering(HF, h')*inv_sum;              % blurred in both orientations
        
    HGF  = GF - allF;                                        % horizontal wavelet plane
    GHF  = HF - allF;                                        % vertical wavelet plane   

    % apply grouplet transform and ECSF to horizontal and vertical wavelet
    % planes:
    gHGF = DHT(HGF', 2, nu_0, mode, s, window_sizes)';
    gGHF = DHT(GHF, 2, nu_0, mode, s, window_sizes);

    % save weighted horizontal and vertical wavelet planes:
    w{s,1}(:,:,1) = gHGF;
    w{s,1}(:,:,2) = gGHF;
    
    % Create diagonal wavelet plane:
    DF = image - (allF + HGF + GHF);

    % apply grouplet transform and ECSF to diagonal wavelet plane at 45 and
    % 135 degrees:
    gDF = DHT(DF, 2, nu_0, mode, s, window_sizes);    
    gDF = gDF + DHT(DF', 2, nu_0, mode, s, window_sizes)';        

    % save weighted diagonal wavelet plane:
    w{s,1}(:,:,3) = gDF;
   
    % Downsample residual image:
    image = allF(1:2:m,1:2:m);

    % set residual data to ones:
    c{s,1} = ones(size(image));
end

end

function rec = DHT(Signal, orientation, nu_0, mode, s, window_sizes)
% Apply grouplet decomposition to Signal, weight grouplet
% coefficients using ECSF, reconstruct Signal using interpolation
% 
% outputs:
%   rec: weighted reconstructed Signal
%
% inputs:
%   Signal: signal to be decomposed.
%   orientation: orientation of wavelet plane corresponding to Signal
%   nu_0: offset of peak spatial frequency for CSF in case of image
%   resizing
%   mode: type of channel i.e. colour or intensity
%   window sizes: window sizes for computing relative contrast; suggested 
%   value of [17 37]

[M N] = size(Signal);
maxJ  = log2(N);
r     = Signal;

rec = 0;
for j=1:maxJ    
    [r d] = blockMatching(r,j);

    % calculate normalized center contrast:
    Zctr = norm_center_contrast(d, orientation, window_sizes);

    % return alpha values:
    alpha = generate_csf(Zctr, s, nu_0, mode);
    
    % reconstruction using interpolation:
    if ~isempty(alpha)
        rec = rec + imresize(alpha,[M N]);    
    end
end

end
