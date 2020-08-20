%Simple code for Topographic ICA of images
%Aapo Hyvärinen, for the book Natural Image Statistics

function W=tica(Z,topoxdim,topoydim,neighbourhoodsize)

global convergencecriterion

%------------------------------------------------------------
% input parameters settings
%------------------------------------------------------------
%
% Z                    : whitened data
%
% topoxdim             : number of components or each row of topographic grid
%
% topoydim             : number of components or each col of topographic grid
%
% maxiter = 100, say   : max number of iterations in algorithm
%
% neighbourhoodsize= 1...5   : neighbourhood size for ISA


%------------------------------------------------------------
% Initialize algorithm
%------------------------------------------------------------

%create matrix where the i,j-th element is one if they are neighbours
%and zero if not
H = neighbourhoodmatrix(topoxdim,topoydim,neighbourhoodsize);

%create random initial value of W, and orthogonalize it
W = orthogonalizerows(randn(topoxdim*topoydim,size(Z,1))); 

%read sample size from data matrix
N = size(Z,2);

%Initial step size
mu = 1;

%------------------------------------------------------------
% Start algorithm
%------------------------------------------------------------

writeline('Doing topographic ICA. Iteration count: ')

iter = 0;
notconverged = 1;

while notconverged & (iter<2000) %max 2000 iterations

  iter=iter+1;
  
  %print iterations left
  writenum(iter);

  %-------------------------------------------------------------
  % Gradient step for topographic ICA
  %-------------------------------------------------------------  

    % Compute separately estimates of independent components to speed up 
    Y=W*Z; 
    
    %compute local energies K
    K=H*Y.^2; 

    % This is nonlinearity corresponding to generalized exponential density
    % (note minus sign)
    epsilon=0.1;
    gK = -(epsilon+K).^(-0.5);        

    % Calculate convolution with neighborhood
    F=H*gK;
    
    % This is the basic gradient
    grad = (Y.*F)*Z'/N;

    % project gradient on tangent space of orthogonal matrices (optional)
    grad=grad-W*grad'*W;

    %store old value
    Wold = W;

    % do gradient step
    W = W + mu*grad;

    % Orthogonalize rows or decorrelate estimated components
    W = orthogonalizerows(W);


    % Adapt step size every tenth step
    if rem(iter,1)==0 | iter==1
      
      changefactor=4/3;

      % Take different length steps
      Wnew1 = Wold + 1/changefactor*mu*grad;
      Wnew2 = Wold + changefactor*mu*grad;
      Wnew1=orthogonalizerows(Wnew1);
      Wnew2=orthogonalizerows(Wnew2);
      
      % Compute objective function values
      J1=-sum(sum(sqrt(epsilon+H*(Wnew1*Z).^2)));
      J2=-sum(sum(sqrt(epsilon+H*(Wnew2*Z).^2)));
      J0=-sum(sum(sqrt(epsilon+H*(W*Z).^2)));

      % Compare objective function values, pick step giving minimum
      if J1>J2 & J1>J0
	% Take shorter step because it is best
	mu = 1/changefactor*mu;
	%%fprintf(' [mu=%f] ',mu);
        W=Wnew1;
      elseif J2>J1 & J2>J0
	% Take longer step because it is best
	mu = changefactor*mu;
	%%fprintf(' [mu=%f] ',mu);
        W=Wnew2;
      end
  end

  % Check if we have converged
  if norm(grad,'fro') < convergencecriterion *topoxdim*topoydim; 
             notconverged=0; end

end %of gradient iterations loop











    
