function [ size_out ] = size_coords( x,y,size_in,X,Y )

    [m]=size_in(1); [n]=size_in(2);
    [t,r]=cart2pol(n,m);
    
    [theta,rho]=cart2pol(x,y);
    [THETA,RHO]=cart2pol(X,Y);
    
    %incT=theta./THETA;
    incR=rho./RHO;
    incX=x./X;
    incY=y./Y;
    v=incX./incY;
    
    R=r./incR;
    %T=t./incT; %be careful, theta increment is not proportional
    
    [N,M]=pol2cart(t,R);
    N=N.*v;
    size_out=[M N];
end

