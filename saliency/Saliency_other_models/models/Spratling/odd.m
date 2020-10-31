function x=odd(x,updown)
%function x=odd(x,updown)
%Rounds x to the nearest odd integer
%
%if updown==1 (or is omitted as an input argument) then this function rounds
%to an odd integer greater than or equal to x, otherwise it rounds to an odd integer less than or equal to x

if nargin<2
  updown=1;
end
  
if updown==1
  %returns smallest odd integer value that is greater than or equal to x
  x=ceil(x);
  modindex=find(mod(x,2)==0);
  x(modindex)=x(modindex)+1;
else
  %returns smallest odd integer value that is less than or equal to x
  x=floor(x);
  modindex=find(mod(x,2)==0);
  x(modindex)=x(modindex)-1;
end