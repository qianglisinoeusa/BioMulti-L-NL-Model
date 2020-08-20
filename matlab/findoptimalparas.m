%find optimal freq, orientation and phase for a *linear* rf

function [optx,opty,optfreq,optor,optphase]=findoptimalparas(w,freqvalues,orvalues,phasevalues);

%----------------------------------------------------------------------------
%
% Incoming parameters (all row vectors):
%
% w : linear rf 
%
% freqvalues : the range of different frequencies that will be tried out
%
% orvalues : same for orientations
%
% phasevalues: same for phases
%
% Output parameters are the optimal values of parameters for w: 
%       location (x,y), frequency, orientation, and phase
%
%-----------------------------------------------------------------------------

freqno=size(freqvalues,2);
orno=size(orvalues,2);
phaseno=size(phasevalues,2);
patchsize=sqrt(size(w,2));

%find optimal freq and orientation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%this sums squares of quadrature-phase rf's 
%so we don't need to go through all the different phases

%set initial maximum response
maxresponsesofar=-inf;

%start loop through all parameter values
for freqindex=1:freqno
for orindex=1:orno

%create two gratings in quadrature phase (i.e. sin and cos parts)
%with desired freqs and orientatations
singrating=creategratings(patchsize,freqvalues(freqindex),orvalues(orindex),0);
cosgrating=creategratings(patchsize,freqvalues(freqindex),orvalues(orindex),pi/2);

%compute max response of linear feature over all phases
sinresponse=w*singrating;
cosresponse=w*cosgrating;
response=sinresponse^2+cosresponse^2;

%check if this is max response so far and store values
if response>maxresponsesofar
  maxresponsesofar=response;
  optfreq=freqvalues(freqindex);
  optor=orvalues(orindex);
end 

end
end

%find optimal phase
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%this is separate from optimal freq and orientation to speed up computations

%set initial maximum response
maxresponsesofar=-inf;

%start loop through all parameter values
for phaseindex=1:phaseno

%create a grating with given phase and optimal values for other paras
grating=creategratings(patchsize,optfreq,optor,phasevalues(phaseindex));

%compute response of linear feature
response=w*grating;

%check if this is max response so far and store values
if response>maxresponsesofar
  maxresponsesofar=response;
  optphase=phasevalues(phaseindex);
end 

end %for phaseindex

%find optimal location
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%this does not use a grating so it is naturally separate from the preceding

%create grid of values
grid=[0:patchsize-1]'/patchsize;

%compute envelope in 2D
wsquared=w.^2;
wsquared=wsquared/sum(wsquared);
wenv=reshape(wsquared,[patchsize patchsize]);

%compute central locations as expectations
optx=sum(wenv)*grid;
opty=sum(wenv')*grid;

