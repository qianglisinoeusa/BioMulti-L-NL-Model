function Gabor =create_gabors(Numberofscales,NumbofOrient,Pscale,Pwavelength,Pfiltersize)
% CREATE GABOR FILTERS with "Numberofscales" scales and "NumbofOrient" orientations


if(Numberofscales>length(Pscale))
    error('Number of scale must be between 1 and 16: modify lines 7-9 of "create_gabors.m" if more scales are needed');
end


% Keep a subset of "Numberofscales" scales
clear scales wavelengths filtersizes
scales=Pscale(floor(linspace(1,length(Pscale),Numberofscales)));
wavelengths=Pwavelength(floor(linspace(1,length(Pscale),Numberofscales)));
filtersizes=Pfiltersize(floor(linspace(1,length(Pscale),Numberofscales)));



Numberofscales=length(scales);                                 
angle=0 : pi/NumbofOrient : (NumbofOrient-1)*pi/NumbofOrient ;   
Gabor=cell(Numberofscales,NumbofOrient);      

for (s=1:Numberofscales)         % begin loop for scales
    
clear g
center=(filtersizes(s)+1)/2;     % center of filter
radius= (filtersizes(s)-1)/2;    % radius of filter
sigma=scales(s);                 % scale of filter (sigma of gaussian)
gamma = 0.3;                     % aspect ratio of filter (see the READ_ME.txt file for reference)
lambda= wavelengths(s);          % wavelength of filter

for(r=1:NumbofOrient)            % begin loop for rotations
    theta=angle(r);
    
for (x=-radius:radius)           % creates the filter
    for(y=-radius:radius)
        g(x+center,y+center)=gabor_rad(x,y,sigma,theta,gamma,lambda);
    end
end
g = g - mean(mean(g));
g = g ./ sqrt(sum(sum(g.^2)));
Gabor{s,r}=g(:,:);             % store the filter
end                            % end loop for rotations
end                            % end loop for scales

save('Gabor','Gabor')
end

