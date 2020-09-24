function [filter_w]=fourier_to_wavelet_soft(filter_f,ns,no);

% 
% fourier_to_wavelet_soft computes the filter in the wavelet domain according to a variation of Malo et al. JMO 97  
% (soft because) It computes fourier transforms and inverses through fft and fast wavelet transforms (does not require the explicit matrices) 
% NOTE!: it is assumed the filter is vectorized with no fftshift and column-wise 
%
% [filter_w]=fourier_to_wavelet_soft(filter_f,ns,no);
%
%    Requires * matlabpyrtools
%             * average_deviation_wavelet from BioMultiLayer_L_NL_color
%

csf = filter_f;
N = sqrt(length(csf));
[p,ind]=buildSFpyr(randn(N,N),ns,no-1);
d = length(p);

filtro2D = reshape(filter_f,[N N]);

% Best diagonal filter

%X = 128 + 20*randn(N*N,N*N);
%B = W*real(iF*diag(csf)*F*X);
%A = W*X;
%clear X W F iF

B = zeros(d,N*N);
A = zeros(d,N*N);

%h = waitbar(0,'CSF filting noise in wavelet domain...');
for i=1:N*N
    x = 128 + 20*randn(N,N);
    xf = real( ifft2(filtro2D.*fft2(x)) );
    [p,ind]=buildSFpyr(xf,ns,no-1);
    B(:,i) = p;
    [p,ind]=buildSFpyr(x,ns,no-1);
    A(:,i) = p;
    %waitbar(i/N*N,h)
end
%close(h)

best_filter_f = zeros(d,1);
%h = waitbar(0,'Optimal weights in wavelet domain...');
for i=1:d
    best_filter_f(i) = B(i,:)*pinv(A(i,:));
    %waitbar(i/d,h)
end
%close(h)
filter_w = average_deviation_wavelet(best_filter_f,ind);