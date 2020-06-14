function [f,dfdx] = saturation_f(x,g,xm,epsilon);

%
% SATURATION_F is an element-wise exponential function (saturation). 
% It is good to (1) model the saturation in Wilson-Cowan recurrent networks, 
% and (2) as a crude (fixed) approximation to the luminance-brightness transform. 
%
% This saturation is normalized and modified to have these two good properties:
% (a) Some specific input, xm (e.g. the median, the average) maps into itself: xm = f(xm).
% 
%          f(x) = sign(x)*K*|x|^g  , where the constant K=xm^(1-g)
% 
% (b) Plain exponential is modified close to the origin to avoid the singularity of
%     the derivative of saturating exponentials at zero.
%     This problem is solved by imposing a parabolic behavior below a certain
%     threshold (epsilon) and imposing the continuity of the parabola and its
%     first derivative at epsilon.
%
%         f(x) = sign(x)*K*|x|^g             for |x| > epsilon
%                sign(x)*(a*|x|+b*|x|^2)     for |x| <= epsilon
%
%      with:
%                a = (2-g)*K*epsilon^(g-1)
%                b = (g-1)*K*epsilon^(g-2)
%
% The derivative is (of course) signal dependent:
% 
%         df/dx = g*K*|x|^(g-1)   for |x| > epsilon   [bigger with xm and decreases with signal]
%                 a + 2*b*|x|     for |x| <= epsilon  [bigger with xm and decreases with signal (note that b<0)]
%
% In the end, the slope at the origin depends on the constant xm (bigger for bigger xm). 
% 
% The program gives the function and the derivative. For the inverse see INV_SATURATION_F.M
%
% For vector/matrix inputs x, the vector/matrix with anchor points, xm, has to be the same size as x.
%
% USE:    [f,dfdx] = saturation_f(x,gamma,xm,epsilon);
%
%   x     = n*m matrix with the values 
%   gamma = exponent (scalar)
%   xm    = n*m matrix with the anchor values (in wavelet representations typically anchors will be different for different subbands)
%   epsilon = threshold (scalar). It can also be a matrix the same size as x (again different epsilons per subband, e.g. epsilon = 1e-3*x_average)
%
% EXAMPLE:
%    [f,dfdx] = saturation_f(linspace(-3,3,1001),0.2,ones(1,1001),0.1);
%    figure,subplot(211),plot(linspace(-3,3,1001),f)
%    subplot(212),semilogy(linspace(-3,3,1001),dfdx)
%

s = size(x);

x = x(:);
xm=xm(:);

K = xm.^(1-g);
a = (2-g)*K.*epsilon.^(g-1);
b = (g-1)*K.*epsilon.^(g-2);

p = find(x>0);
pG = find(x > epsilon);
pp = find((x <= epsilon) & (x >= 0));

n = find(x<0);
nG = find(x<-epsilon);
np = find((x > -epsilon) & (x <= 0));

f = x;
dfdx=f;

if isempty(pG)==0
   f(pG) = K(pG).*x(pG).^g;            %        for |x| > epsilon
   dfdx(pG) = g*K(pG).*abs(x(pG)).^(g-1);    %   for |x| > epsilon   [bigger with xm and decreases with signal]
end
if isempty(nG)==0
   f(nG) = - K(nG).*abs(x(nG)).^g;     %        for |x| > epsilon
   dfdx(nG) = g*K(nG).*abs(x(nG)).^(g-1);    %   for |x| > epsilon   [bigger with xm and decreases with signal]
end

if isempty(pp)==0
   f(pp) = (a(pp).*abs(x(pp))+b(pp).*abs(x(pp)).^2);       %        for |x| <= epsilon
   dfdx(pp) = a(pp) + 2*b(pp).*abs(x(pp));   %  for |x| <= epsilon  [bigger with xm and decreases with signal (note that b<0)]
end
   
if isempty(np)==0
   f(np) = -(a(np).*abs(x(np))+b(np).*abs(x(np)).^2);      %        for |x| <= epsilon
   dfdx(np) = a(np) + 2*b(np).*abs(x(np));   %  for |x| <= epsilon  [bigger with xm and decreases with signal (note that b<0)]
end

f=reshape(f,s);
dfdx=reshape(dfdx,s);
