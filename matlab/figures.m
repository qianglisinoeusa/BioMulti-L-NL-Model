%Main file for generating basic experiments for the book
%"Natural Image Statistics" by Hyvarinen, Hurri, and Hoyer.

%To launch this m-file just call it without any input arguments

%SET GLOBAL VARIABLES

%This is path where the figures are saved. Default: same directory. 
global figurepath 
figurepath='Result/';

%These variables decide which sections are run and which are not
%Set all to 1 if you want to run the whole thing
do_basic_ornot=1;
do_pca_ornot=1;
do_ica_ornot=1;
do_isa_ornot=1;
do_tica_ornot=1;
do_overcomplete_ornot=1;

%The following give basic parameters used in the experiments

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

%define default colormap
colormap('gray')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  BASIC STATISTICS SECTION (Section 1.8 and Sections 5.1, 5.2.1, 5.6)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if do_basic_ornot %DO BASIC STATISTICS SECTION ?

%initialize random number generators to get same results each time
initializerandomseeds;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% COMPUTE FILTER OUTPUTS FOR SOME PREDETERMINED FILTERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

writeline('------------------------------------')
writeline('Starting basic statistics section...')
writeline('------------------------------------')

writeline('Creating feature detectors')

%create dirac filter
diracfilter=zeros(patchsize^2,1);
diracfilter(floor(patchsize/2)*patchsize+floor(patchsize/2))=1;
diracfilter=diracfilter/norm(diracfilter);

%create sinusoidalfilter
sinusoidalfilter=creategratings(patchsize,2,pi/2,0);
sinusoidalfilter=sinusoidalfilter/norm(sinusoidalfilter);

%create edge by Gabor function
envelope=0.5; freq=1;
%change the following function to be more standard:
edgefilter=gabor(patchsize,0,0,envelope,freq,'o');
edgefilter=edgefilter/norm(edgefilter);

%print filters

figure, plotrf(diracfilter,1,'diracfilter')
figure, plotrf(sinusoidalfilter,1,'sinusoidalfilter')
figure, plotrf(edgefilter,1,'edgefilter')

%get natural image data
writeline('Sampling data')
X=sampleimages(samplesize,patchsize);
%create another data set with no DC component
writeline('Removing DC component')
XnoDC=removeDC(X);

writeline('Plotting a few natural image patches')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%how many  patches
patchno=20;
indices=randperm(samplesize);
figure, plotrf(XnoDC(:,indices(1:patchno)),5,'natimpatches')



writeline('Computing feature detectors outputs')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%compute outputs for different basic filters
diracoutputs=diracfilter'*X;
sinusoidaloutputs=sinusoidalfilter'*X;
edgeoutputs=edgefilter'*X;
diracoutputs_noDC=diracfilter'*XnoDC;
sinusoidaloutputs_noDC=sinusoidalfilter'*XnoDC;
edgeoutputs_noDC=edgefilter'*XnoDC;

%Define common x-range for histograms in terms of std of the outputs
xrange=[-4.8:.2:4.8];

%compute, plot and save histograms
figure,histogram(diracoutputs,xrange*std(diracoutputs),'dirachistogram')
figure,histogram(sinusoidaloutputs,xrange*std(sinusoidaloutputs),'sinusoidalhistogram')
figure,histogram(edgeoutputs,xrange*std(edgeoutputs),'edgehistogram')
figure,histogram(diracoutputs_noDC,xrange*std(diracoutputs_noDC),'dirachistogram_noDC')
figure,histogram(sinusoidaloutputs_noDC,xrange*std(sinusoidaloutputs_noDC),'sinusoidalhistogram_noDC')
figure,histogram(edgeoutputs_noDC,xrange*std(edgeoutputs_noDC),'edgehistogram_noDC')

writeline('Scatterplot of neighbouring pixels')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%make scatterplot of two neighbouring pixels
plotsamplesize=500;
figure, plot_withbigfont(X(1,1:plotsamplesize),X(2,1:plotsamplesize),'.')
print('-deps',[figurepath,'neighpixelscatterplot.eps'])
figure, plot_withbigfont(XnoDC(1,1:plotsamplesize),XnoDC(2,1:plotsamplesize),'.')
print('-deps',[figurepath,'neighpixelscatterplot_noDC.eps'])


writeline('plot one row of covariance matrix')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%choose reference pixel
x0=patchsize/2;
y0=x0;
%compute index of reference pixel
index=x0+patchsize*(y0-1);

%compute covariances with DC
Cx0y0_withDC=zeros(patchsize^2,1);
meanofX=mean(X')';
for t=1:samplesize
  Cx0y0_withDC=Cx0y0_withDC+(X(index,t)-meanofX(index))*(X(:,t)-meanofX);
end
%compute standard deviation of pixel
stdofpixel_withDC=std(X(index,:));
%normalize to correlation coefficients
Cx0y0_withDC=Cx0y0_withDC/stdofpixel_withDC^2/samplesize;

figure, plotrf(Cx0y0_withDC,1,'rowofcovmatrix_withDC')

figure, plot_withbigfont([0:patchsize-1]-x0,Cx0y0_withDC(patchsize*(y0-1):patchsize*y0-1))
axis([-patchsize/2+1,patchsize/2-1,0,1])

print('-deps',[figurepath,'rowofcovmatrix1d_withDC.eps'])

%compute covariances without DC
Cx0y0_noDC=zeros(patchsize^2,1);
for t=1:samplesize
  Cx0y0_noDC=Cx0y0_noDC+XnoDC(index,t)*XnoDC(:,t); %here, means are zero
end
%compute standard deviation of a pixel (does not matter which one)
stdofpixel_noDC=std(XnoDC(index,:));
%normalize to correlation coefficients
Cx0y0_noDC=Cx0y0_noDC/stdofpixel_noDC^2/samplesize;

figure,plotrf(Cx0y0_noDC,1,'rowofcovmatrix_noDC')

figure,plot_withbigfont([0:patchsize-1]-x0,Cx0y0_noDC(patchsize*(y0-1):patchsize*y0-1))
axis([-patchsize/2+1,patchsize/2-1,-.1,1])
print('-deps',[figurepath,'rowofcovmatrix1d_noDC.eps'])

writeline('Analysing anisotropy')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%choose the different radii for which the autocorrelation is shown 
radii=[patchsize/2.1,patchsize/4,patchsize/8];
%choose the different angle values
angles=[0:.05:pi];

%Compute the pixel coordinates corresponding to all possible combinations
%of those radii and angles
xcoord=ceil(x0+radii'*cos(angles));
ycoord=ceil(y0+radii'*sin(angles));

%take the autocorrelation values computed previously
clf
h=axes;
axis([0,pi,-.2,.5])
hold on
for radiusindex=1:size(radii,2)
h2=plot(angles,Cx0y0_noDC(patchsize*(xcoord(radiusindex,:)-1)+ycoord(radiusindex,:)));
set(h2,'LineWidth',2)
end

set(h,'fontSize',25)
set(h,'XTick',[0,1.57,3.14])
print('-deps',[figurepath,'anisotropy.eps']);
hold off

end % OF BASIC STATISTICS SECTION


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  PCA SECTION (most of Chapter 5)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if do_pca_ornot %DO PCA SECTION ?

%to save memory, get rid of some things computed in previous section(s)
clear X XnoDC Xnorm

writeline('------------------------------------')
writeline('Starting PCA section...')
writeline('------------------------------------')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% POWER SPECTRUM ANALYSIS (Section 5.6)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

writeline('Power spectra experiments')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%read two whole natural images
im1 = double(imread(['data/5.tiff']));
im2 = double(imread(['data/11.tiff']));

%Take only square part of the image
im1=im1(:,257:512); 
im2=im2(:,200:455);

%im=im-mean(im(:));

%compute 2D Fourier transform of the image 1
im1f=fft2(im1);
%take phase
im1f_phase=angle(im1f);
%take amplitude
im1f_abs=abs(im1f);

%same for image 2
im2f=fft2(im2);
im2f_phase=angle(im2f);
im2f_abs=abs(im2f);

%compute image with phase from image 1 and power from image 2
%(take only real part in the end because 
% numerical errors give small imaginary parts)
%Also note that "i" may be redefined as an index
im_ph1_abs2=real(ifft2(im2f_abs.*exp(sqrt(-1)*im1f_phase)));

%same other way round
im_ph2_abs1=real(ifft2(im1f_abs.*exp(sqrt(-1)*im2f_phase)));

clf
figure, imagesc(im1)
axis off
print('-deps',[figurepath,'natim1.eps']);
figure, imagesc(im2)
axis off
print('-deps',[figurepath,'natim2.eps']);
figure, imagesc(im_ph1_abs2)
axis off
print('-deps',[figurepath,'natim_ph1_abs2.eps']);
figure, imagesc(im_ph2_abs1)
axis off
print('-deps',[figurepath,'natim_ph2_abs1.eps']);


writeline('Computing power spectra')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%size of image
fouriersize=size(im1,2);

%Partly adapted from code by Bruno Olshausen
im1f=fftshift(fft2(im1));
im2f=fftshift(fft2(im2));
im1_pf=abs(im1f).^2;
im2_pf=abs(im2f).^2;
f=-fouriersize/2:fouriersize/2-1;

%plot 2d log of power spectrum
figure,imagesc(f,f,log10(im1_pf))
axis xy
print('-deps',[figurepath,'powerspectrumnatim2d.eps']);

%plot log-log 1-d crosssection of power spectrum
Pf1=rotavg(im1_pf);
Pf2=rotavg(im2_pf);
freq=[0:fouriersize/2]';
logPf1=log10(Pf1(2:fouriersize/2));
logPf2=log10(Pf2(2:fouriersize/2));
logfreq=log10(freq(2:fouriersize/2));
figure, plot_withbigfont(logfreq,logPf1,logfreq,logPf2);
print('-deps',[figurepath,'powerspectrumnatim1dlog.eps']);

%this plot is not included in book (and ugly): 
%original frequencies instead of logs
figure, plot_withbigfont(sqrt(freq),Pf1,sqrt(freq),Pf2);
print('-deps',[figurepath,'powerspectrumnatim1d.eps']);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% COMPUTE PCA (Sections 5.2.2 and 5.2.3)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%initialize random number generators to get same results each time
initializerandomseeds;

writeline('Computing 1st and 100th PCs from ten random samples ')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
print('-deps',[figurepath,'pcad.eps'])


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
print('-deps',[figurepath,'antialiasing.eps'])
figure, plotrf(nyquistgrating1,1,'nyquistgrating')
figure, plotrf(checkergrating,1,'checkergrating')


end % OF PCA SECTION


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  ICA / SPARSE CODING SECTION (Chapters 6, 7, 9)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if do_ica_ornot % DO ICA SECTION ?


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% COMPUTE BASIC ICA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%to save memory, get rid of some things computed in previous section(s)
clear X XnoDC Xnorm

writeline('------------------------------------')
writeline('Starting ICA section...')
writeline('------------------------------------')

%initialize random number generators to get same results each time
initializerandomseeds;

%Sample data and preprocess
writeline('Sampling data')
X=sampleimages(samplesize,patchsize);
writeline('Removing DC component')
X=removeDC(X);
writeline('Doing PCA and whitening data')
[V,E,D]=pca(X);
Z=V(1:rdim,:)*X;


writeline('Starting computation of some individual ICs.')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

threeICs=zeros(patchsize^2,3);

for i=1:3

%Find one sparse component using FastICA
W=ica(Z,1);
%Transform back to original space from whitened space
Worig = W*V(1:rdim,:);
threeICs(:,i)=Worig';

end % of for i

figure, plotrf(threeICs,3,'threeicas') %This is Figure 6.5


writeline('Starting complete ICA. ')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This computes the main results for chapters 6 and 7. (See below for chapter 9)

W=ica(Z,rdim); 
%transform back to original space from whitened space
Wica = W*V(1:rdim,:);
%Compute A using pseudoinverse (inverting canonical preprocessing is tricky)
Aica=pinv(Wica);

figure,plotrf(Aica,plotcols,'icaA')  %This is Figure 7.3

figure,plotrf(Wica',plotcols,'icaW') %This is Figure 6.6

%compute outputs of simple cells
Sica=Wica*X;

writeline('Synthesizing images using the ICA model')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%how many synthesized patches
synthno=20;

%take samples of each component with random indices, so they are independent
synthcoeffs1=zeros(rdim,synthno);
for i=1:rdim
indices=ceil(rand(1,synthno)*samplesize);
synthcoeffs1(i,:)=Sica(i,indices);
end
icasynth1=Aica*synthcoeffs1;
plotrf(icasynth1,5,'icasynth1')

%alternatively, use Laplace distribution
synthcoeffs2=log(rand(rdim,synthno)).*sign(randn(rdim,synthno));
icasynth2=Aica*synthcoeffs2;
plotrf(icasynth2,5,'icasynth2')



writeline('Computing optimal sparseness measures') %Figure 7.6
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%find most and mid kurtotic components
kurtoses=mean(Sica'.^4);
[dummy,midkurtindex]=min(floor(size(kurtoses,2)/2)+1);
[dummy,maxkurtindex]=max(kurtoses);
s_kurtmid=Sica(midkurtindex,:);
s_kurtmax=Sica(maxkurtindex,:);

% Compute all the stuff for mid kurtosis feature

%set range
xrange=[-5:.33:5];
%find positive half of this range
[dummy,zeroind]=min(abs(xrange));
positivehalfindices=[zeroind:size(xrange,2)];

%compute pdf and its log, plot log
counts=hist(s_kurtmid,xrange);
logpdfmid=log(counts+1);
plot_withbigfont(xrange,logpdfmid);
print('-deps',[figurepath,'optlogpdfmid.eps']);
%plot h_i 
plot_withbigfont(xrange(positivehalfindices).^2,logpdfmid(positivehalfindices));
print('-deps',[figurepath,'opthimid.eps']);
%plot derivative of log-pdf, only in the mid range where there's data
logpdfdermid=diff(logpdfmid);
goodrange=[ceil(size(xrange,2)*.25):ceil(size(xrange,2)*.75)];
plot_withbigfont(xrange(goodrange)+.33/2,logpdfdermid(goodrange))
print('-deps',[figurepath,'optlogpdfdermid.eps'])
%plot the feature as well
plotrf(Wica(midkurtindex,:)',1,'opthimidfeature')

% Compute all the same stuff for maximum kurtosis feature

%compute pdf and its log, plot log
counts=hist(s_kurtmax,xrange);
logpdfmax=log(counts+1);
plot_withbigfont(xrange,logpdfmax);
print('-deps',[figurepath,'optlogpdfmax.eps'])
%plot h_i 
plot_withbigfont(xrange(positivehalfindices).^2,logpdfmax(positivehalfindices))
     print('-deps',[figurepath,'opthimax.eps']);
%plot derivative of log-pdf, only in the mid range where there's data
logpdfdermax=diff(logpdfmax);
goodrange=[ceil(size(xrange,2)*.25):ceil(size(xrange,2)*.75)];
plot_withbigfont(xrange(goodrange)+.33/2,logpdfdermax(goodrange))
print('-deps',[figurepath,'optlogpdfdermax.eps'])
%plot the feature as well
plotrf(Wica(maxkurtindex,:)',1,'opthimaxfeature')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ANALYZE ICA RESULTS (Section 6.4.2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

writeline('Analyzing tuning of ICA features')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%set number of different values for the grating parameters used in 
%computing tuning curves and optimal parameters
freqno=50; %how many frequencies
orno=50; %how many orientations
phaseno=20; %how many phases
%compute the used values for the orientation angles and frequencies
orvalues=[0:orno-1]/orno*pi;
freqvalues=[0:freqno-1]/freqno*patchsize/2;
phasevalues=[0:phaseno-1]/phaseno*2*pi;

%initialize optimal values
ica_optfreq=zeros(1,rdim);
ica_optor=zeros(1,rdim);
ica_optphase=zeros(1,rdim);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%ANALYZE TUNING FOR ALL SIMPLE CELLS
%i is index of simple cell
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i=1:rdim;

writenum(rdim-i)

%find optimal parameters for the i-th linear feature estimated by ICA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[optxtmp,optytmp,optfreqtmp,optortmp,optphasetmp]=findoptimalparas(Wica(i,:),freqvalues,orvalues,phasevalues);

ica_optx(i)=optxtmp;
ica_opty(i)=optytmp;
ica_optfreq(i)=optfreqtmp;
ica_optor(i)=optortmp;
ica_optphase(i)=optphasetmp;

%compute responses when phase is changed for an ICA feature
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

phaseresponse=zeros(1,phaseno);

for phaseindex=1:phaseno

%create new grating with many phases
grating=creategratings(patchsize,ica_optfreq(i),ica_optor(i),phasevalues(phaseindex));

%compute response
phaseresponse(phaseindex)=Wica(i,:)*grating;

end %for phaseindex

%normalize
phaseresponse=phaseresponse/max(abs(phaseresponse));

%compute responses when freq is changed for an ICA feature
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%note: this is responses to "drifting gratings", 
%i.e. we cannot use optimal phase but have to recompute optimal phase
%separately. in practice, this is done by computing fourier amplitude
%for given frequency and orientation

freqresponse=zeros(1,freqno);

for freqindex=1:freqno

%create new grating with many freqs 
singrating=creategratings(patchsize,freqvalues(freqindex),ica_optor(i),0);
cosgrating=creategratings(patchsize,freqvalues(freqindex),ica_optor(i),pi/2);

%compute response
sinresponse=Wica(i,:)*singrating;
cosresponse=Wica(i,:)*cosgrating;
freqresponse(freqindex)=sqrt(sinresponse^2+cosresponse^2);

end %for freqindex

%compute responses when orientation is changed for an ICA feature
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

orresponse=zeros(1,orno);

for orindex=1:orno

%create new grating with many phases
singrating=creategratings(patchsize,ica_optfreq(i),orvalues(orindex),0);
cosgrating=creategratings(patchsize,ica_optfreq(i),orvalues(orindex),pi/2);

%compute response
sinresponse=Wica(i,:)*singrating;
cosresponse=Wica(i,:)*cosgrating;
orresponse(orindex)=sqrt(sinresponse^2+cosresponse^2);

end %for orindex

%plot and save results for the first simple cells estimated by ICA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if i<=10


%plot phase tuning curve
plot_withbigfont(phasevalues,phaseresponse)
axis([min(phasevalues),max(phasevalues),-1,1]);
print('-deps',[figurepath,'icasel' num2str(i) 'c.eps']),

%plot freq tuning curve
plot_withbigfont(freqvalues,freqresponse)
print('-deps',[figurepath,'icasel' num2str(i) 'a.eps']),

%plot orientation tuning curve
plot_withbigfont(orvalues,orresponse);
print('-deps',[figurepath,'icasel' num2str(i) 'b.eps']);

end %of if


end %for i loop through simple cells




%PLOT ORIENTATION-FREQUENCY TILING FOR ICA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%add some randomness so that if points are on top of each other,
%they will be seen as a thick point
plot_withbigfont(ica_optor+randn(size(ica_optor))*.01,ica_optfreq,'*');
print('-deps',[figurepath,'icatilingscatter.eps']);

histogram(ica_optor,[.25:.25:pi-.25],'icaortiling')
histogram(ica_optfreq,[min(freqvalues)+.5:1:max(freqvalues)-.5],'icafreqtiling')



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  EXAMINE CONTRAST GAIN CONTROL METHODS (Chapter 9)           %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


writeline('Plot histograms of correlations of nonlinear transforms')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%set range for histograms
xrange=[-.145:.01:.345];

%compute correlations for different nonlinearities and plot histograms
abscorrs=computecorrelations(abs(Sica));
histogram(abscorrs,xrange,'abscorrelations')
energycorrs=computecorrelations(Sica.^2);
histogram(energycorrs,xrange,'energycorrelations')
threscorrs=computecorrelations(abs(Sica)>1);
histogram(threscorrs,xrange,'threscorrelations')
signcorrs=computecorrelations(sign(Sica));
histogram(signcorrs,xrange,'signcorrelations')
cubiccorrs=computecorrelations(Sica.^3);
histogram(cubiccorrs,xrange,'cubiccorrelations')



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%DO ICA ON VARIANCE-NORMALIZED DATA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%DO VERY BASIC VARIANCE NORMALIZATION:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Xnorm=variancenormalize(X);

writeline('-----------------------------------------')
writeline('Starting ICA on variance-normalized data.')

[Vnorm,Enorm,Dnorm]=pca(Xnorm);
Xnormwhite=Vnorm(1:rdim,:)*Xnorm;
W_cgc=ica(Xnormwhite,rdim); 

%transform back to original space 
Wica_cgc = W_cgc*Vnorm(1:rdim,:);
Aica_cgc = pinv(Wica_cgc);

%old:
%Wica_cgc = W_cgc*Vnorm(1:rdim,:)*Wica;

plotrf(Aica_cgc,plotcols,'icaA_cgc')

plotrf(Wica_cgc',plotcols,'icaW_cgc')


%ANALYZE CHANGE IN DEPENDENCIES AND DISTRIBUTIONS DUE TO VARIANCE NORMALIZATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%compute values of independent components after variance normalization
S_cgc=W_cgc*Xnormwhite;

%compute correlations for different nonlinearities and plot histograms
abscorrs=computecorrelations(abs(S_cgc));
histogram(abscorrs,xrange,'abscorrelations_cgc')
energycorrs=computecorrelations(S_cgc.^2);
histogram(energycorrs,xrange,'energycorrelations_cgc')
threscorrs=computecorrelations(abs(S_cgc)>1);
histogram(threscorrs,xrange,'threscorrelations_cgc')

%compute kurtoses and plot histograms
icakurtoses=mean(Sica'.^4)-3;
histogram(icakurtoses,[.5:.5:15],'icakurtoses')
icakurtoses_cgc=mean(S_cgc'.^4)-3;
histogram(icakurtoses_cgc,[.5:.5:15],'icakurtoses_cgc')

end % OF ICA SECTION



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  ISA SECTION (Chapter 10)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if do_isa_ornot % DO ISA SECTION?

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% COMPUTE ISA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%to save memory, get rid of some things computed in previous section(s)
clear Sica Z X XnoDC Xnorm

writeline('------------------------------------')
writeline('Starting ISA section')
writeline('------------------------------------')

%initialize random number generators to get same results each time
initializerandomseeds;

%choose size of subspace used in all these simulations
subspacesize=4;

writeline('Sampling images')
X=sampleimages(samplesize,patchsize);

writeline('Removing DC component')
X=removeDC(X);

writeline('Variance-normalizing data')
X=variancenormalize(X);

writeline('Doing PCA and whitening data')
[V,E,D]=pca(X);
Z=V(1:rdim,:)*X;

writeline('Starting computation of one independent subspace.  ')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

W=isa_est(Z,subspacesize,subspacesize);

Worig = W*V(1:rdim,:);

plotrf(Worig',subspacesize,'oneisa')

writeline('Showing different combinations of ISA features.')

combno=20; %how many combinations

isacombinations=zeros(patchsize^2,combno);

for i=1:combno

coefficients=randn(1,subspacesize);
isacombinations(:,i)=(coefficients*Worig)';

end

plotrf(isacombinations,10,'isacombinations')

writeline('Starting computation of whole ISA.')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

W=isa_est(Z,rdim,subspacesize);

%transform back to original space from whitened space
Wisa = W*V(1:rdim,:);
%basis vectors computed using pseudoinverse
Aisa=pinv(Wisa);

writeline('Ordering subspaces according to sparseness.')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Compute outputs
Sisa=Wisa*X;

%order subspaces according to square correlations
energies=Sisa.^2;
%compute likelihood inside subspaces
subspaceenergies=sum(reshape(energies,[subspacesize,rdim/subspacesize,samplesize]),1);
subspaceLstmp=-sqrt(subspaceenergies);
subspaceLs=sum(subspaceLstmp,3);
[values,ssorder]=sort(subspaceLs(1,:,:)','descend');
tmporder=((ssorder-1)*ones(1,subspacesize)*subspacesize+ones(rdim/subspacesize,1)*[1:subspacesize])';
componentorder=tmporder(:);

%do re-ordering of features
Wisa=Wisa(componentorder,:);
Aisa=Aisa(:,componentorder);

%PLOT ISA FEATURES
%%%%%%%%%%%%%%%%%%
%plotted only at this stage because we first wanted to order them

plotrf(Aisa,plotcols,'isaA')
plotrf(Wisa',plotcols,'isaW')


writeline('Synthesizing images using the ISA model')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%how many synthesized patches
synthno=20;

%take the norm of each subspace with random indices, so they are independent 
synthcoeffs=zeros(rdim,synthno); 
for i=1:rdim/subspacesize 
indices=ceil(rand(1,synthno)*samplesize);
subspacenorm=sqrt(sum(Sisa((i-1)*subspacesize+1:i*subspacesize,indices).^2));
%create coefficients with random directions inside the subspaces
%a simple way of doing this is to take gaussian independent variables, 
%and dividing them by their norm
randdir=randn(subspacesize,synthno);
randdir=randdir./(ones(subspacesize,1)*sqrt(sum(randdir.^2)));
%finally, multiply these two, so you get something with the prescribed norm, 
%and random angles
synthcoeffs((i-1)*subspacesize+1:i*subspacesize,:)=randdir.*(ones(subspacesize,1)*subspacenorm);
end

isasynth=Aisa*synthcoeffs;
plotrf(isasynth,5,'isasynth')



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ANALYZE ISA RESULTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%ENERGY CORRELATION ANALYSIS OF ISA RESULTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%compute correlations of squares
CC2=corrcoef(Sisa'.^2);

%plot part of this matrix (clipped at 0.3) and save
imagesc(min(CC2(1:subspacesize*20,1:subspacesize*20),0.3))
print('-deps',[figurepath,'isaenergycorrimage.eps']);

%compute correlations inside subspaces
insidecorrsorig=zeros(rdim*subspacesize-rdim,1);
%take out diagonal part (i.e.\ variances of squares)
CC2od=CC2-diag(diag(CC2));
for i=1:rdim/subspacesize
%take out a piece of correlations matrix which corresponds to this subspace
piece=CC2od(1+(i-1)*subspacesize:i*subspacesize,1+(i-1)*subspacesize:i*subspacesize);
%for histogram, take all elements in that part
insidecorrsorig(1+(i-1)*subspacesize^2:i*subspacesize^2)=piece(:);
end
%take out elements which are exactly zero, i.e. the diagonal elements
indices=find(insidecorrsorig~=0);
insidecorrs=insidecorrsorig(indices);

%plot histogram of correlations inside subspaces
histogram(insidecorrs,[-.1:.02:.5],'isainsidecorrs')

%compute correlations between components in different subspaces
outsidecorrsorig=CC2;
for i=1:rdim/subspacesize
%set to zero those entries which correspond to components in the same subspace
outsidecorrsorig(1+(i-1)*subspacesize:i*subspacesize,1+(i-1)*subspacesize:i*subspacesize)=zeros(subspacesize);
end
%choose only those elements which are not zero, i.e. not in same subspace
indices=find(outsidecorrsorig~=0);
outsidecorrs=outsidecorrsorig(indices);
%plot histogram of energy correlations between components in different subspaces
histogram(outsidecorrs,[-.1:.02:.5],'isaoutsidecorrs')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
writeline('Analyzing tuning of ISA features')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%set number of different values for the grating parameters used in 
%computing tuning curves and optimal parameters
orno=50; %how many orientations
freqno=50; %how many frequencies 
phasenolinear=50; %how many phases for linear RF's
phasenosubs=20; %how many phases for subspaces RF's 
                %(should be smaller because computation more complicated)
%compute the used values for the orientation angles and frequencies
orvalues=[0:orno-1]/orno*pi;
freqvalues=[0:freqno-1]/freqno*patchsize/2; 
phasevalueslinear=[0:phasenolinear-1]/phasenolinear*2*pi;
phasevaluessubs=[0:phasenosubs-1]/phasenosubs*2*pi;

%initialize values of optimal parameters
isa_optor=zeros(1,rdim/subspacesize);
isa_optfreq=zeros(1,rdim/subspacesize);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%ANALYZE TUNNING FOR ALL COMPLEX CELLS
%i is index of comple cell
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i=1:rdim/subspacesize;

writenum(rdim/subspacesize-i)

%FIRST, FIND OPTIMAL PARAMETERS FOR CELL OF INDEX i
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%initialize the maximum response of cell i found so far
maxresponsesofar=-inf;

%start loop through all parameter values fro frequency and orientation
for freqindex=1:freqno
for orindex=1:orno

%we take sum over responses for different phases to simulate drifting grating
%initialize the variable which holds the sum for cell i
response=0; 

for phaseindex=1:phasenosubs

%create a grating with desired freqs and orientations and phases
grating=creategratings(patchsize,freqvalues(freqindex),orvalues(orindex),phasevaluessubs(phaseindex));

%compute linear responses of RF's underlying feature subspace 
linearresponses=Wisa((i-1)*subspacesize+1:i*subspacesize,:)*grating;
%add their energy to energy found for this cell with other phases
response=response+sqrt(sum(linearresponses.^2)); 

end %of for phaseindex 

%check if this is max response so far and store values
if response>maxresponsesofar
  maxresponsesofar=response;
  isa_optfreq(i)=freqvalues(freqindex);
  isa_optor(i)=orvalues(orindex);
  %%isa_optphase(i)=phasevaluessubs(phaseindex);
end 


end %of for freqindex
end %of for orindex

%NEXT, COMPUTE RESPONSE TUNING CURVES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%compute ISA responses when phase is changed
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

phaseresponse=zeros(1,phasenosubs);
linearresponses=zeros(subspacesize,phasenosubs);

for phaseindex=1:phasenosubs

%create new grating with many phases
grating=creategratings(patchsize,isa_optfreq(i),isa_optor(i),phasevaluessubs(phaseindex));

%compute response
linearresponses(:,phaseindex)=Wisa((i-1)*subspacesize+1:i*subspacesize,:)*grating;
phaseresponse(phaseindex)=sqrt(sum(linearresponses(:,phaseindex).^2));

end %for phaseindex 

%normalize to simplify visualization of invariance
phaseresponse=phaseresponse/max(phaseresponse);


%compute ISA responses when freq is changed
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

freqresponse=zeros(1,freqno);

for freqindex=1:freqno
for phaseindex=1:phasenosubs

%create new grating with different freq
grating=creategratings(patchsize,freqvalues(freqindex),isa_optor(i),phasevaluessubs(phaseindex));

%Compute response of complex cell
linearresponses=Wisa((i-1)*subspacesize+1:i*subspacesize,:)*grating;
%take sum of responses over different phases
freqresponse(freqindex)=freqresponse(freqindex)+sqrt(sum(linearresponses.^2));

end %for phaseindex
end %for freqindex



%compute ISA responses when orientation is changed
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

orresponse=zeros(1,orno);

for orindex=1:orno
for phaseindex=1:phasenosubs

%create new grating with different orientation and phase
grating=creategratings(patchsize,isa_optfreq(i),orvalues(orindex),phasevaluessubs(phaseindex));

%compute response of complex cell
linearresponses=Wisa((i-1)*subspacesize+1:i*subspacesize,:)*grating;
%take sum of responses over different phases
orresponse(orindex)=orresponse(orindex)+sqrt(sum(linearresponses.^2));

end %for phaseindex
end %for orindex

%plot and save tuning curves and for the first complex cells estimated by ISA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if i<=10  


%plot phase tuning curve
plot_withbigfont(phasevaluessubs,phaseresponse)
axis([min(phasevaluessubs),max(phasevaluessubs),0,1])
print('-deps',[figurepath,'isasel' num2str(i) 'c.eps']),

%plot freq tuning curve
plot_withbigfont(freqvalues,freqresponse)
print('-deps',[figurepath,'isasel' num2str(i) 'a.eps']),

%plot orientation tuning curve
plot_withbigfont(orvalues,orresponse)
print('-deps',[figurepath,'isasel' num2str(i) 'b.eps']),

end %of if i

end %for i loop through complex cells



%PLOT ORIENTATION-FREQUENCY TILING FOR ISA (for subspaces)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

histogram(isa_optor,[.25:.25:pi-.25],'isaortiling')
histogram(isa_optfreq,[min(freqvalues)+.5:1:max(freqvalues)-.5],'isafreqtiling')

%PLOT CORRELATIONS OF PARAMETERS INSIDE SUBSPACES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

writeline('Computing optimal parameters for all linear rfs in ISA')

for i=1:rdim;

writenum(rdim-i)

%find optimal parameters for the i-th linear feature estimated by ISA
[optxtmp,optytmp,optfreqtmp,optortmp,optphasetmp]=findoptimalparas(Wisa(i,:),freqvalues,orvalues,phasevalueslinear);

isa_linear_optx(i)=optxtmp;
isa_linear_opty(i)=optytmp;
isa_linear_optfreq(i)=optfreqtmp;
isa_linear_optor(i)=optortmp;
isa_linear_optphase(i)=optphasetmp;

end %of for i

%plot correlations for two consecutive linear features: 
%they are in the same subspace if subspacesize is even

freqs1=isa_linear_optfreq(1:2:rdim);
freqs2=isa_linear_optfreq(2:2:rdim);
plot_withbigfont(freqs1,freqs2,'*');
print('-deps',[figurepath,'isacorrfreq.eps']);

ors1=isa_linear_optor(1:2:rdim);
ors2=isa_linear_optor(2:2:rdim);
plot_withbigfont(ors1,ors2,'*');
axis([0,pi,0,pi])
print('-deps',[figurepath,'isacorror.eps']);

phases1=isa_linear_optphase(1:2:rdim);
phases2=isa_linear_optphase(2:2:rdim);
plot_withbigfont(phases1,phases2,'*');
axis([0,2*pi,0,2*pi])
print('-deps',[figurepath,'isacorrphase.eps']);

loc1=isa_linear_optx(1:2:rdim);
loc2=isa_linear_optx(2:2:rdim);
plot_withbigfont(loc1,loc2,'*');
print('-deps',[figurepath,'isacorrloc.eps']);


end % OF ISA SECTION




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  TOPOGRAPHIC ICA SECTION (Chapter 11)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if do_tica_ornot % DO TOPOGRAPHIC ICA SECTION?


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% COMPUTE TOPOGRAPHIC ICA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%to save memory, get rid of some things computed in previous section(s)
clear Sica Sisa Z X XnoDC Xnorm

writeline('------------------------------------')
writeline('Starting TOPOGRAPHIC ICA section...')
writeline('------------------------------------')

%initialize random number generators to get same results each time
initializerandomseeds;

%define topographic parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%the "radius" of the neighbourhood for computing local energies
neighbourhoodsize=2;
topoxdim=plotcols; %size of grid, x axis
topoydim=rdim/topoxdim; %size of grid, y axis

writeline('Sampling images')
X=sampleimages(samplesize,patchsize);

writeline('Removing DC component')
X=removeDC(X);

writeline('Variance-normalizing data')
X=variancenormalize(X);

writeline('Doing PCA and whitening data')
[V,E,D]=pca(X);
Z=V(1:rdim,:)*X;

writeline('Starting topographic ICA. ')
W=tica(Z,topoxdim,topoydim,neighbourhoodsize);

%transform back to original space from whitened space
Wtica = W*V(1:rdim,:);

%rotate grid so that low-frequency blob are in the middle
%-- for better visualization only
%lower frequency W_i's have smallest norm so we can use that
%and force the W_i with smallest norm to be in the middle
[dummy,blobindex]=min(sum(Wtica'.^2));
centerindex=round(topoxdim/2)+round(topoydim/2)*topoxdim;
if centerindex<blobindex
  Wtica=Wtica([(blobindex-centerindex+1):(topoxdim*topoydim),1:(blobindex-centerindex)],:);
end
if centerindex>blobindex
  Wtica=Wtica([(topoxdim*topoydim+blobindex-centerindex+1):(topoxdim*topoydim),1:(topoxdim*topoydim+blobindex-centerindex)],:);
end

%compute basis vectors using pseudoinverse
Atica=pinv(Wtica);

%plor results
plotrf(Atica,plotcols,'ticaA')
plotrf(Wtica',plotcols,'ticaW')


writeline('Synthesizing images using the TICA model')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%how many synthesized patches
synthno=20;

%generate underlying gaussian variables
gaussianvariables=randn(rdim,synthno);
%generate higher-order independent component
U=randn(rdim,synthno).^4;
%mix them to give variance variables
H=neighbourhoodmatrix(topoxdim,topoydim,neighbourhoodsize);
variancevariables=H*U;
%finally, generate data by multiplying these two:
synthcoeffs=variancevariables.*gaussianvariables;

ticasynth=Atica*synthcoeffs;
plotrf(ticasynth,5,'ticasynth')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ANALYZE TOPOGRAPHIC ICA RESULTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

writeline('Analyzing tuning of TICA features')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%set number of different values for the grating parameters used in 
%computing tuning curves and optimal parameters
orno=50; %how many orientations
freqno=50; %how many frequencies
phaseno=50; %how many phases
%compute the used values for the orientation angles and frequencies
orvalues=[0:orno-1]/orno*pi;
freqvalues=[0:freqno-1]/freqno*patchsize/2; %perhaps too many!
phasevalues=[0:phaseno-1]/phaseno*pi*2;

%initialize values
tica_optor=zeros(1,rdim);
tica_optfreq=zeros(1,rdim);
tica_optphase=zeros(1,rdim);


%analyze tuning for linear receptive fields
%i is index of receptive field

for i=1:rdim;

writenum(rdim-i)

%find optimal parameters for the i-th linear feature estimated by TICA
[optxtmp,optytmp,optfreqtmp,optortmp,optphasetmp]=findoptimalparas(Wtica(i,:),freqvalues,orvalues,phasevalues);

tica_optx(i)=optxtmp;
tica_opty(i)=optytmp;
tica_optfreq(i)=optfreqtmp;
tica_optor(i)=optortmp;
tica_optphase(i)=optphasetmp;

end


%PLOT ORIENTATION-FREQUENCY TILING FOR TICA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

plot_withbigfont(tica_optor,tica_optfreq,'*');
print('-deps',[figurepath,'ticatiling.eps']);

histogram(tica_optor,[.25:.25:pi-.25],'ticaortiling')
histogram(tica_optfreq,[min(freqvalues)+.5:1:max(freqvalues)-.5],'ticafreqtiling')


%PLOT CORRELATIONS OF NEIGHBOURS IN TICA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
plot_withbigfont(tica_optfreq(1:rdim-1),tica_optfreq(2:rdim),'*');
print('-deps',[figurepath,'ticacorrfreq.eps']);

plot_withbigfont(tica_optor(1:rdim-1),tica_optor(2:rdim),'*');
axis([0,pi,0,pi])
print('-deps',[figurepath,'ticacorror.eps']);

plot_withbigfont(tica_optphase(1:rdim-1),tica_optphase(2:rdim),'*');
axis([0,2*pi,0,2*pi])
print('-deps',[figurepath,'ticacorrphase.eps']);

plot_withbigfont(tica_optx(1:rdim-1),tica_optx(2:rdim),'*');
print('-deps',[figurepath,'ticacorrx.eps']);

plot_withbigfont(tica_opty(1:rdim-1),tica_opty(2:rdim),'*');
print('-deps',[figurepath,'ticacorry.eps']);

%PLOT GLOBAL PARAMETER MAPS
%%%%%%%%%%%%%%%%%%%%%%%%%%%

colormap('jet')
imagesc(reshape(tica_optphase,[topoxdim,topoydim])'); 
print('-depsc',[figurepath,'ticamapphase.eps']);
imagesc(reshape(tica_optor,[topoxdim,topoydim])'); 
print('-depsc',[figurepath,'ticamapor.eps']);
colormap('gray')
imagesc(reshape(tica_optfreq,[topoxdim,topoydim])'); 
print('-depsc',[figurepath,'ticamapfreq.eps']);
imagesc(reshape(tica_optx,[topoxdim,topoydim])'); 
print('-depsc',[figurepath,'ticamapx.eps']);
imagesc(reshape(tica_opty,[topoxdim,topoydim])'); 
print('-depsc',[figurepath,'ticamapy.eps']);

end %OF TICA SECTION




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  OVERCOMPLETE BASIS SECTION (Section 13.1)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


if do_overcomplete_ornot % DO OVERCOMPLETE BASIS SECTION ?

%to save memory, get rid of some things computed in previous section(s)
clear X XnoDC Xnorm

writeline('-----------------------------------------')
writeline('Starting overcomplete basis section...')
writeline('-----------------------------------------')

%initialize random number generators to get same results each time
initializerandomseeds;

%Take smaller windows to avoid excessive computation
patchsize_oc=floor(patchsize/2);

%Don't reduce dimension as much as usual so that the overcomplete basis
%is also overcomplete in the sense of having more features than pixels
rdim_oc=floor(rdim/2);

%Sample data and preprocess
writeline('Sampling data')
X=sampleimages(samplesize,patchsize_oc);

writeline('Removing DC component')
X=removeDC(X);

writeline('Doing PCA and whitening data')
[V,E,D]=pca(X);
Z=V(1:rdim_oc,:)*X;

clear X %memory may be getting quite used up here!

writeline('Starting overcomplete basis estimation. ')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%how many times overcomplete as a function of PCA dimension
oc_factor=4; 

W=overcompletebasis(Z,oc_factor*rdim_oc); 

%transform back to original space from whitened space
Woc = W*V(1:rdim_oc,:);

plotrf(Woc(1:12*16,:)',12,'overcompleteW')


end %OF OVERCOMPLETE SECTION


