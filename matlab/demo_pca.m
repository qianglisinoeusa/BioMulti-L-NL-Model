%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% COMPUTE PCA (Sections 5.2.2 and 5.2.3)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%initialize random number generators to get same results each time
initializerandomseeds;

writeline('Computing 1st and 100th PCs from ten random samples ')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%sample size, i.e. how many image patches. Book value: 50000
samplesize=50000; 
%patchsize in most experiments. Book value: 32
patchsize=32;
%Number of features or weight vectors in one column in the big plots
%Book value: 16
plotcols=16; 
%Number of features computed, i.e. PCA dimension in big experiments
%Book value: plotcols*16, or 256
rdim=plotcols*16; 

%Choose "small" value which determines when the change in estimate is so small
%that algorithm can be stopped. 
%This is related to the proportional change allowed for the features
%Book value: 1e-4, i.e. accuracy must be of the order of 0.01%
global convergencecriterion  
convergencecriterion=1e-4;

%How many times are the 1st and 100th principal components computed
pcano=10;

firstpcs=zeros(patchsize^2,pcano);
hundredthpcs=zeros(patchsize^2,pcano);

for t=1:pcano

%sample images
writeline('  Sampling data')
X=sampleimages(samplesize,patchsize);
writeline('  Removing DC component')
X=removeDC(X);

%do PCA
writeline('  Computing PCA number '); writenum(t)
[V,E,D]=pca(X);

%store computed principal components
firstpcs(:,t)=E(:,1);
hundredthpcs(:,t)=E(:,100); 


end

figure, plotrf(firstpcs,pcano,'firstpcs')

figure, plotrf(hundredthpcs,pcano,'hundredthpcs')


writeline('Computing one whole PCA')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

writeline('  Sampling data')
X=sampleimages(samplesize,patchsize);
writeline('  Removing DC component')
X=removeDC(X);
writeline('  Doing whole PCA')
[V,E,D]=pca(X);

%Plot almost whole PCA
figure,plotrf(E(:,1:320),plotcols,'pca')

d=diag(D);
figure,plot_withbigfont(1:patchsize^2-1,log10(d(1:patchsize^2-1)));


writeline('Synthesizing images using the PCA model')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%how many synthesized patches
synthno=20;

%generate gaussian coefficients with the variances of the principal components
synthcoeffs=real(diag(sqrt(diag(D))))*randn(patchsize^2,synthno);
pcasynth=E*synthcoeffs;
figure, plotrf(pcasynth,5,'pcasynth')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ANALYSIS OF CHOICE OF PCA DIMENSION (Figure 5.12)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

writeline('Computing dimension reduction needed to downsample properly')

highestwantedgrating1=creategratings(patchsize,patchsize/4,0,pi/2);
highestwantedcumenergy1=cumsum((E'*highestwantedgrating1).^2)*100;
highestwantedgrating2=creategratings(patchsize,patchsize/4,pi/4,pi/2);
highestwantedcumenergy2=cumsum((E'*highestwantedgrating2).^2)*100;
nyquistgrating1=creategratings(patchsize,patchsize/2,0,pi/2);
nyquistcumenergy1=cumsum((E'*nyquistgrating1).^2)*100;
nyquistgrating2=creategratings(patchsize,patchsize/2,pi/4,pi/2);
nyquistcumenergy2=cumsum((E'*nyquistgrating2).^2)*100;
checkergrating=creategratings(patchsize,patchsize/2*sqrt(2),pi/4,pi/2);
checkercumenergy=cumsum((E'*checkergrating).^2)*100;
percentagecounter=[1:patchsize^2]/patchsize^2*100;
figure, plot_withbigfont(percentagecounter,nyquistcumenergy1,'--',percentagecounter,nyquistcumenergy2,'--',percentagecounter,highestwantedcumenergy1,'-',percentagecounter,highestwantedcumenergy2,'-',percentagecounter,checkercumenergy,':')
figure, plotrf(nyquistgrating1,1,'nyquistgrating')
figure, plotrf(checkergrating,1,'checkergrating')

