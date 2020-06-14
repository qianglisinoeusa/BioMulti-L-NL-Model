%Simple code for ISA of images
%Aapo Hyvärinen, for book Natural Image Statistics
%Note: this code is actually mostly the same as the code for topographic ICA
%      since the only different is in the structure of the pooling


function W=isa_est(Z,n,subspacesize)
%This is called isa_est to avoid confusion with built-in matlab function 'isa'

global convergencecriterion

%------------------------------------------------------------
% input parameters settings
%------------------------------------------------------------
%
% Z                      : whitened data
%
% n = 60..windowsize^2-1 : number of linear components to be estimated
%                          (must be divisible by subspacesize)
%
% subspacesize= 2...10   : subspace size 


%------------------------------------------------------------
% Initialize algorithm
%------------------------------------------------------------

%create matrix where the i,j-th element is one if they are in same subspace
%and zero if in different subspaces
ISAmatrix=subspacematrix(n,subspacesize);

%create random initial value of W, and orthogonalize it
W = orthogonalizerows(randn(n,size(Z,1))); 

%read sample size from data matrix
N=size(Z,2);

%Initial step size
mu = 1;

%------------------------------------------------------------
% Start algorithm
%------------------------------------------------------------

writeline('Doing ISA. Iteration count: ')

iter = 0;
notconverged = 1;

while notconverged & (iter<2000) %maximum of 2000 iterations

  iter=iter+1;
  
  %print iteration count
  writenum(iter);


  %-------------------------------------------------------------
  % Gradient step for ISA
  %-------------------------------------------------------------  


    % Compute separately estimates of independent components to speed up 
    Y=W*Z; 
    
    %compute energies of subspaces
    K=ISAmatrix*Y.^2; 

    % This is nonlinearity corresponding to generalized exponential density
    % (note minus sign)
    epsilon=0.1;
    gK = -(epsilon+K).^(-0.5);        

    % Calculate product with subspace indicator matrix
    F=ISAmatrix*gK;
    
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


    % Adapt step size every step, or every n-th step? remove this?
    if rem(iter,1)==0 | iter==1

      %How much do we want to change the step size? Choose this factor
      changefactor=4/3;

      % Take different length steps
      Wnew1 = Wold + 1/changefactor*mu*grad;
      Wnew2 = Wold + changefactor*mu*grad;
      Wnew1=orthogonalizerows(Wnew1);
      Wnew2=orthogonalizerows(Wnew2);
      
      % Compute objective function values
      J1=-sum(sum(sqrt(epsilon+ISAmatrix*(Wnew1*Z).^2)));
      J2=-sum(sum(sqrt(epsilon+ISAmatrix*(Wnew2*Z).^2)));
      J0=-sum(sum(sqrt(epsilon+ISAmatrix*(W*Z).^2)));

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

   %Here, we could have an stopping criterion based on the change in W
   %but we prefer to just take the fixed number of iterations
   %because assessing the convergence of ISA is not straightforward
   %due to the indeterminacy of rotations inside subspaces

   %check convergence
   if norm(grad,'fro') < convergencecriterion * n; notconverged=0; end

end %of gradient iterations loop





