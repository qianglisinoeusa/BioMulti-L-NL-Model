% ----------------------------------------
% 
% FastICA
%
% z: data after whitening, nDim x nSamples
% B: transposed demixing matrix, nDim x nComponents
%    Columns are the filters (in whitened space)
%
% ----------------------------------------
  
function B = fastICA(z,nComponents,fileName,maxIter)

% initialize
    dim = size(z,1);
    B = randn(dim,nComponents);  
    N = size(z,2);
    
    for k=1:maxIter
        % This is tanh but faster than matlabs own version
        hypTan = 1 - 2./(exp(2*(z'*B))+1);
        
        % This is the fixed-point step
        B = z*hypTan/N - repmat(mean(1-hypTan.^2),dim,1).*B;
        B = B*real((B'*B)^(-0.5));
        
        if mod(k,20)==0
            fprintf(['Saving at iteration : ' num2str(k) '\n'])
            save(fileName,'B','k')
        end
            
    end

    