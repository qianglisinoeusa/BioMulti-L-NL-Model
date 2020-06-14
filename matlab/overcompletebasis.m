%Estimate overcomplete basis from image patches
%based on energy-based approach and score matching
%Aapo Hyvärinen for the book Natural Image Statistics
%Note: this algorithm is much slower than the others!

function W=overcompletebasis(Z,n)

global convergencecriterion

%------------------------------------------------------------
% input parameters settings
%------------------------------------------------------------
%
% Z   : whitened image patch data
%
% n   : number of basis vectors to be estimated


%------------------------------------------------------------
% Initialize algorithm 
%------------------------------------------------------------

%random initial W with unit row norms
W = randn(n,size(Z,1)); 
W=W./(sqrt(sum(W'.^2))'*ones(1,size(W,2)));

alpha=2*ones(1,n);

%read sample size
N=size(Z,2);

%------------------------------------------------------------
% Start algorithm
%------------------------------------------------------------

writeline('Estimating overcomplete basis. Iteration count: ')

iter = 0;
notconverged = 1;

while notconverged & (iter<5000) %maximum of 5000 iterations

  iter=iter+1;

  %print iteration count
  writenum(iter);
  
  % Store old value
  Wold=W;        

  %------------------------------------------------------------
  % Compute filter outputs
  %------------------------------------------------------------

  Y=W*Z; 

  %-------------------------------------------------------------
  % Overcomplete PoE step
  %-------------------------------------------------------------  

  %compute g(Y) and its first derivative
  gY=-tanh(W*Z);
  gpY=-(1-gY.^2);

  %A useful matrix
  Egigj=gY*gY'/N;


  %compute gradient row-by-row

  grad=zeros(size(W));

  for i=1:n
      %compute a complicated part of gradient of w_i
      coeff=alpha.*((W(i,:)*W')/(W(i,:)*W(i,:)'));
      v=(coeff*gY).*gpY(i,:);

      %compute gradient of w_i
      %this uses the identity tanh''(y)=2 tanh'(y) tanh(y)
      grad(i,:)=2*alpha(i)*W(i,:)*mean(gpY(i,:))...
                + alpha(i)*(W(i,:)*W(i,:)')*((2*gY(i,:).*gpY(i,:)+v)*Z')/N...
                + alpha(i)*(alpha.*Egigj(i,:))*W;

      %project gradient to tangent space of unit norm constraint
      grad(i,:)=grad(i,:)-W(i,:)*grad(i,:)'*W(i,:);
  end
 
  %Do gradient step for W
  %Here, no attempt is made to optimize the step size
  stepsize=.1;
  W=W-stepsize*grad;

  %normalize rows of W to unit norm
  W=W./(sqrt(sum(W'.^2))'*ones(1,size(W,2)));

  %estimate alphas: 
  M=Egigj.*(W*W');
  b=sum(W'.^2).*mean(gpY');
  gradalpha=alpha*M+b;
  alpha=alpha-0.01*gradalpha;

  % Check if converged by comparing norm of gradient with small value
  if norm(grad,'fro') < convergencecriterion * n ; notconverged=0; end

end %of iterations loop

