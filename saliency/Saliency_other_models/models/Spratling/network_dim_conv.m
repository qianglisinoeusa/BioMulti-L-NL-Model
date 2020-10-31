function [Y,R,E,Ytrace,Rtrace,Etrace]=network_dim_conv(w,X,iterations,A,downsample)
% w = a cell array of size {N,M}, where N is the number of distinct neuron types
%     and M is the number of input channels. Each element {i,j} of the cell
%     array is a 2-dimensional matrix (a convolution mask) specifying the
%     synaptic weights for neuron type i's RF in input channel j.
% X = a cell array of size {M}. Each element {j} of the cell array is a
%     2-dimensional matrix specifying the bottom-up input to that channel j of
%     the current processing stage (external inputs targetting the error
%     nodes). If inputs arrive from more than one source, then each source
%     provides a different cell in the array. Input values that change over
%     time can be presented using a 3-d matrix, where the 3rd dimension is time.
% Y = a cell array of size {N}. Each element {i} of the cell array is a
%     2-dimensional matrix specifying prediction node activations for type i
%     neurons.
% R = a cell array of size {M}. Each element {j} of the cell array is a
%     2-dimensional matrix specifying the reconstruction of the input channel j.
%     These values are also the top-down feedback that should modulate
%     activity in preceeding processing stages of a hierarchy.
% E = a cell array of size {M}. Each element {j} of the cell array is a
%     2-dimensional matrix specifying the error in the reconstruction of the
%     input channel j.


[a,b,z]=size(X{1});
[nMasks,nChannels]=size(w);
[c,d]=size(w{1,1});
if nargin<3, iterations=25; end
if nargin<4 || isempty(A), initA=1; else initA=0; end
if nargin<5, downsample=ones(1,nMasks); end

%set parameters
epsilon=1e-9;
psi=5000;
epsilon1=0.0001; %>0.001 this becomes significant compared to y and hence
%produces sustained responses and more general suppression
epsilon2=500*epsilon1*psi;%this determines scaling of initial transient response
%(i.e. response to linear filters).
eta=1;

%try to speed things up
if exist('convnfft')==2 && (max(c,d)>=50-max(a,b) || (min(c,d)>10 && min(a,b)>10))
  conv_fft=1;%use fft version of convolution for large images and/or masks
else
  conv_fft=0;%use standard conv2 function for smaller images and/or masks
end
%also convert data to single precision
%for convnfft single is always faster than double
%for conv2 single is faster than double if image is larger than mask
for j=1:nChannels
  X{j}=single(X{j});
end

%normalize weights and initialise outputs
for i=1:nMasks
  %initialise prediction neuron outputs to zero
  Y{i}=zeros(a,b,'single');
  if initA, A{i}=zeros(a,b,'single'); end

  %calculate normalisation values by taking into account all weights
  %contributing to each RF type
  sumVal=0;
  maxVal=0;
  for j=1:nChannels
    w{i,j}=single(w{i,j});
    sumVal=sumVal + sum(sum(w{i,j}));
    maxVal=max(maxVal,max(max(w{i,j})));
  end
  sumVal=sumVal./psi;
  maxVal=maxVal./psi;
  
  %apply normalisation to calculate feedforward and feedback weight values.
  %Note: FF weights are flipped versions, so that conv2 can be used to apply the
  %filtering
  for j=1:nChannels
    wFF{i,j}=fliplr(flipud(w{i,j}))./(epsilon+sumVal);
    wFB{i,j}=w{i,j}./(epsilon+maxVal);
  end
end

fprintf(1,'dim_conv(%i): ',conv_fft);
%iterate to determine steady-state response
for t=1:iterations
  fprintf(1,'.%i.',t);
  
  %update error units
  for j=1:nChannels
    R{j}=zeros(a,b,'single');%reset reconstruction of input
    if conv_fft==1
      for i=1:nMasks
        R{j}=R{j}+convnfft(Y{i},wFB{i,j},'same');%sum reconstruction over
        %each RF type
      end
    else
      for i=1:nMasks
        R{j}=R{j}+conv2(Y{i},wFB{i,j},'same');%sum reconstruction over
        %each RF type
      end
    end
    E{j}=X{j}(:,:,min(t,z))./(epsilon2+R{j});%calc reconstruction error
    if nargout>5
      Etrace{j}(:,:,t)=E{j}.*psi;%record response over time
    end
    if nargout>4
      Rtrace{j}(:,:,t)=R{j}./psi;%record response over time
    end
  end
  
  %update outputs
  for i=1:nMasks
    input=0;
    if conv_fft==1
      for j=1:nChannels
        input=input+convnfft(E{j},wFF{i,j},'same');%sum input to prediction
        %node from each channel
      end
    else
      for j=1:nChannels
        input=input+conv2(E{j},wFF{i,j},'same');%sum input to prediction
        %node from each channel
      end
    end
    Y{i}=(epsilon1+Y{i}).*input.*(1+eta.*A{i});%modules prediction node response by input
    Y{i}=max(0,Y{i});%ensure no negative values creep in!
    for ds=1:downsample(i)-1
      Y{i}(ds:downsample(i):a,:)=0;
      Y{i}(:,ds:downsample(i):b)=0;
    end
    if nargout>3
      Ytrace{i}(:,:,t)=Y{i};%record response over time
    end
  end
  
end
for j=1:nChannels
  R{j}=R{j}./psi;
  E{j}=E{j}.*psi;
end
disp(' ');

