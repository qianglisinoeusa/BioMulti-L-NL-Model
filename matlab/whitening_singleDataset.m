% ----------------------------------------
%
% [z eigenResults] =whitening_singleDataset(x,dim)
% 
% z: whitened x, sorted so that component
%    which had largest variance comes first
%
% Eigenvalue decomposition of covariance matrix: 
% sorted so that largest eigenvalues and their eigenvectors come first
% 
% eigenResults.U: all eigenvectors
% eigenResults.d: all eigenvalues
% eigenResults.Wm: whitening matrix with dimension reduction
% eigenResults.m: the mean
% eigenResults.dim: retained dimensions for Wm
%
% ----------------------------------------

function [z eigenResults] =whitening_singleDataset(x,dim)
        
    % remove mean
    m = mean(x,2);
    
    x = bsxfun(@plus,x,-m);
    
    % whitening
    C = cov(x');
    [U D] = eig(C);

    % sort so that largest variance comes first
    [d index] = sort(diag(D),1,'descend');
    U = U(:,index);
    myU = U(:,1:dim);
    myd = d(1:dim);
        
    Wm = diag(myd.^(-0.5))*myU';
    z = Wm*x;
          
    eigenResults.U = U;
    eigenResults.d = d;
    eigenResults.Wm = Wm;
    eigenResults.m = m;
    eigenResults.dim = dim;