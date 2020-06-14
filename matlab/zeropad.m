function imm = zeropad(im,blocksize);

% ZEROPAD completes with zeros till the size of the image 
% is multiple of blocksize
%
% im2 = zeropad(im,blocksize);
%

s = size(im);
r = s(1);
c = s(2);

modr = mod(r,blocksize);
modc = mod(c,blocksize);

if length(s)==2
   imm = zeros(r+modr,c+modc);
   imm(1:r,1:c)=im;
else
   imm = zeros(r+modr,c+modc,3);
   imm(1:r,1:c,:)=im;
end