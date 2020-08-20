function L2 = L2_layer(L1,L2stepsize,maxneigh)
% L2 LAYER: Local max of L1 layer over local position at all scales and all orientation

% INPUTS: 
% Layer L1

% OUTPUT:
% Layer L2: reduced version of L1 by selecting maxima in spatial
% neighborhoods of size maxneigh in step size "L2stepsize" and over 2 adjacent scales.


[a b c d]=size(L1);
Numberofscales=c;
NumbofOrient=d;

L2=zeros(a,b,c-1,d);    
    
for (s=1:Numberofscales-1)                            % loop on scales  
    for (o=1:NumbofOrient)                           % loop on orientation
m1= minmaxfilt(L1(:,:,s,o),[maxneigh(s) maxneigh(s)],'max','same');
m2= minmaxfilt(L1(:,:,s+1,o),[maxneigh(s) maxneigh(s)],'max','same');  
L2(:,:,s,o)=max(m1,m2);        
    end
end
L2=L2(1:L2stepsize:end,1:L2stepsize:end,:,:);


end