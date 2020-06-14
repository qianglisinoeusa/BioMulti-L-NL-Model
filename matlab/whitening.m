% ----------------------------------------
%
% [z1 z2 eigenResults] =whitening(x1,x2,dim)
%
% z1: whitened x1 (e.g. dataA)
% z2: whitened x2 (e.g. dataD)
%
% Eigenvalue decomposition of covariance matrix: 
% sorted so that largest eigenvalues and their eigenvectors come first
% 
% eigenResults.U1,2: all eigenvectors
% eigenResults.d1,2: all eigenvalues
% eigenResults.Wm1,2: whitening matrix with dimension reduction
% eigenResults.m1,2: the mean
% eigenResults.dim: retained dimensions for Wm
%
% ----------------------------------------

function [z1 z2 eigenResults] =whitening(x1,x2,dim)
        
    % remove mean
    m1 = mean(x1,2);
    m2 = mean(x2,2);
    
    x1 = bsxfun(@plus,x1,-m1);
    x2 = bsxfun(@plus,x2,-m2);
    
    % whitening
    C1 = cov(x1');
    C2 = cov(x2');
    [U1 D1] = eig(C1);
    [U2 D2] = eig(C2);
    
    % sort so that largest variance comes first
    [d1 index] = sort(diag(D1),1,'descend');
    U1 = U1(:,index);
    myU1 = U1(:,1:dim);
    myd1 = d1(1:dim);
    
    [d2 index] = sort(diag(D2),1,'descend');
    U2 = U2(:,index);
    myU2 = U2(:,1:dim);
    myd2 = d2(1:dim);
    
    Wm1 = diag(myd1.^(-0.5))*myU1';
    Wm2 = diag(myd2.^(-0.5))*myU2';
    z1 = Wm1*x1;
    z2 = Wm2*x2;
          
    eigenResults.U1 = U1;
    eigenResults.U2 = U2;
    eigenResults.d1 = d1;
    eigenResults.d2 = d2;
    eigenResults.Wm1 = Wm1;
    eigenResults.Wm2 = Wm2;
    eigenResults.m1 = m1;
    eigenResults.m2 = m2;
    eigenResults.dim = dim;