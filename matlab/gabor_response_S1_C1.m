writeline('-----------------------------------------')
writeline('-----------------------------------------')
writeline('-----------------------------------------')
writeline('Gabor function for simulating V1 function')
writeline('-----------------------------------------')
writeline('-----------------------------------------')

Pscale=[2.8 3.6 4.5 5.4 6.3 7.3 8.2 9.2 10.2 11.3 12.3 13.4 14.6 15.8 17.0 18.2];          
Pwavelength=[3.5 4.6 5.6 6.8 7.9 9.1 10.3 11.5 12.7 14.1 15.4 16.8 18.2 19.7 21.2 22.8];   
Pfiltersize=[7:2:37]; 

% Image paramaters
shortestside=140;                                    % Images are rescaled so the shortest is equal to "shortestside" keeping aspect ratio

% Layer 1 parameters
NumbofOrient=12;                                    % Number of spatial orientations for the Gabor filter on the first layer 
Numberofscales=8;                                   % Number of scales for the Gabor filter on the first laye: must be between 1 and 16.
                                                    % Modify line 7-9 of create_gabors.m to increase to more than than 16 scales
% Layer 2 layer parameters
maxneigh=floor(8:length(Pscale)/Numberofscales:8+length(Pscale));  % Size of maximum filters (if necessary adjust according Gabor filter sizes)
L2stepsize=4;                                                      % Step size of L2 max filter (downsampling)
inhi=0.5;                                                          % Local inhibition ceofficient (competition) between Gabor outputs at each position.
                                                                   % Coefficient is between 0 and 1
% Layer 3 (Learning parameters)
NumOfSampledImages=40;                               % Number of images per category from which the dictionary is learned
featureperimage=1;                                   % Number of features learned from each image
feature_spatial_sizes=[4:4:16];                      % List of possible spatial sizes for features: Maximum value must be less than (shortestside/L2stepsize)
feature_scaledepth=7;                                % Scale depth of dictionary features (i.e L3 filters): Must be less than Numberofscales

% Layer 4 paramaters
pooling_radii=[0.12];    
                                                  % Modify line 7-9 of create_gabors.m to increase to more than than 16 scales

%% LOAD AND DISPLAY GABOR FILTERS
Gabor=create_gabors(Numberofscales,NumbofOrient,Pscale,Pwavelength,Pfiltersize);
displaygabors(Gabor)
figure,
displayFFTgabors(Gabor)
figure,
displayMeshFFTgabors(Gabor)

A=getsampleimage;
% L1 LAYER  (NORMALIZED DOT PRODUCT OF GABORS FILTERS ON LOCAL PATCHES OF IMAGE "A" AT EVERY POSSIBLE LOCATIONS AND SCALES)
L1 = L1_layer(Gabor, A);

figure()
ca=0;
for (i=1:size(Gabor,1))
    for (j=1:size(Gabor,2))
        ca=ca+1;
        subplot(size(Gabor,1),size(Gabor,2),ca)
        imagesc(L1(:,:,ca)); axis off; colormap(gray);
    end
end


% L2 LAYER: LOCAL MAX POOLING OF L1 LAYER OVER LOCAL POSITION AT ALL SCALES AND ALL ORIENTATIONS
% THE MAXIMUM POOLING SLIDING WINDOW SIZES ARE CONTAINED IN "maxneigh" AND "L2stepsize" INDICATES THE CORRESPONDING STEPSIZE 
L2 = L2_layer(L1,L2stepsize,maxneigh);

figure()
ca=0;
for (i=1:(size(Gabor,1)-1))
    for (j=1:size(Gabor,2))
        ca=ca+1;
        subplot((size(Gabor,1))-1,size(Gabor,2),ca)
        imagesc(L2(:,:,ca)); axis off; colormap(gray);
    end
end


