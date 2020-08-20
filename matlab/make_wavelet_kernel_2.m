function [W,ind,indices_si] = make_wavelet_kernel_2(N,ns,no,tw,ns_not_used,resid)

% MAKE_WAVELET_KERNEL_2 makes steerable pyramid convolution kernel for N*N images
% not using "M" high frequency scales. If resid = 0 the high frequency
% residual is not used either.
% In that way, if x = image(:), the wavelet transform is w = W*x; and W is 
% not huge since we removed the high frequencies...  
% The program also returns the reduced "ind" matrix to use the
% matlabPyrTools display routines (note one has to add zeros in the -removed- 
% high frequency residual before using showSpyr).
%
% W = make_wavelet_kernel_2(N,ns,no,tw,M,resid);
%

[w,ind] = buildSFpyr(rand(N,N),ns,no-1,tw);

indices_no = [];
for i=1:length(ind(:,1))
    indices = pyrBandIndices(ind,i);
    if resid==1
       if (i > 1) & (i < 2 + ns_not_used*no)
          indices_no = [indices_no indices];
       end    
    else
       if i < 2 + ns_not_used*no
          indices_no = [indices_no indices];
       end
    end
end
indices_tot = 1:length(w);
indices_si = setdiff(indices_tot,indices_no);

W=zeros(length(w)-length(indices_no),N*N);
for i=1:N*N
    delta = zeros(N*N,1);
    delta(i)=1;
    [wd,ind]=buildSFpyr(reshape(delta,N,N),ns,no-1,tw);
    wd = wd(indices_si);
    W(:,i) = wd;
    %i;
end
if resid==1
   ind = [ind(1,:);ind(ns_not_used*no+2:end,:)]; 
else
   ind = [ind(ns_not_used*no+2,:);ind(ns_not_used*no+2:end,:)];
end