% rotavg.m - function to compute rotational average of (square) array
% by Bruno Olshausen
% 
% function f = rotavg(array)
%
% array can be of dimensions N x N x M, in which case f is of 
% dimension NxM.  N should be even.
%

function f = rotavg(array)

[N N M]=size(array);

[X Y]=meshgrid(-N/2:N/2-1,-N/2:N/2-1);

[theta rho]=cart2pol(X,Y);

rho=round(rho);
i=cell(N/2+1,1);
for r=0:N/2
  i{r+1}=find(rho==r);
end

f=zeros(N/2+1,M);

for m=1:M

  a=array(:,:,m);
  for r=0:N/2
    f(r+1,m)=mean(a(i{r+1}));
  end
  
end
