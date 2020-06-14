function y = gabor_rad(x,y,sigma,theta, gamma,lambda)
% Gabor filter output at position (x,y) with parameters:
% sigma  = scale
% theta  = rotation in radians
% gamma  = aspect ratio
% lambda = wavelength

y= exp( -(  (x*cos(theta)+y*sin(theta))^2  +  gamma^2*(-x*sin(theta)+y*cos(theta))^2  ) / (2*sigma^2) )...
    *cos( (2*pi/lambda) * (x*cos(theta)+y*sin(theta)) );

end

