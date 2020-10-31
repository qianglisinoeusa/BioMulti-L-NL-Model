function [ X,Y ] = resize_coords( x,y,size_in,size_out )
    
    [m]=size_in(1); [n]=size_in(2);
    d=hypot(m,n);
    [M]=size_out(1); [N]=size_out(2);
    D=hypot(M,N);
    
    
    %resize rho (hypotenuse)
    [theta,rho]=cart2pol(x,y);
    
    factor=rho./d;
    RHO=factor.*D;
    
    [X,Y]=pol2cart(theta,RHO);
    
    %unequal size proportions, apply variability M/N
    v=m./n;
    V=M./N;
    var=v./V;
    Y=Y.*var;
end

