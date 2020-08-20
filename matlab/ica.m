%Simple code for ICA of images
%Aapo Hyvärinen, for the book Natural Image Statistics

function W=ica(Z,n)

global convergencecriterion

%------------------------------------------------------------
% input parameters settings
%------------------------------------------------------------
%
% Z                     : whitened image patch data
%
% n = 1..windowsize^2-1 : number of independent components to be estimated


%------------------------------------------------------------
% Initialize algorithm
%------------------------------------------------------------

%create random initial value of W, and orthogonalize it
W = orthogonalizerows(randn(n,size(Z,1))); 

%read sample size from data matrix
N=size(Z,2);

%------------------------------------------------------------
% Start algorithm
%------------------------------------------------------------

writeline('Doing FastICA. Iteration count: ')

iter = 0;
notconverged = 1;

while notconverged & (iter<2000) %maximum of 2000 iterations

  iter=iter+1;
  
  %print iteration count
  writenum(iter);
  
  % Store old value
  Wold=W;        

  %-------------------------------------------------------------
  % FastICA step
  %-------------------------------------------------------------  

    % Compute estimates of independent components 
    Y=W*Z; 
 
    % Use tanh non-linearity
    gY = tanh(Y);
    
    % This is the fixed-point step. 
    % Note that 1-(tanh y)^2 is the derivative of the function tanh y
    W = gY*Z'/N - (mean(1-gY'.^2)'*ones(1,size(W,2))).*W;    
    
    % Orthogonalize rows or decorrelate estimated components
    W = orthogonalizerows(W);

  % Check if converged by comparing change in matrix with small number
  % which is scaled with the dimensions of the data
  if norm(abs(W*Wold')-eye(n),'fro') < convergencecriterion * n; 
        notconverged=0; end

end %of fixed-point iterations loop








