% Create neighbourhood structure for TICA, expressed in a single matrix

function H=neighbourhoodmatrix(xdim,ydim,size)

% n is dimension of data
% size is longest distance of neighbours (i.e. neighbourhood size) e.g. 1 or 2

% This will hold the neighborhood function entries
H = zeros(xdim*ydim,xdim*ydim);

% Step through nodes one at a time to build the matrix
ind=0;
for y=1:ydim

  for x=1:xdim
     
    ind=ind+1;


    % Rectangular neighbors
    [xn,yn] = meshgrid( (x-size):(x+size), (y-size):(y+size) );
    xn = reshape(xn,[1 (size*2+1)^2]);
    yn = reshape(yn,[1 (size*2+1)^2]);
      
      % Cycle round to create torus
      i = find(yn<1); yn(i)=yn(i)+ydim;
      i = find(yn>ydim); yn(i)=yn(i)-ydim;
      i = find(xn<1); xn(i)=xn(i)+xdim;
      i = find(xn>xdim); xn(i)=xn(i)-xdim;
      
    % Set neighborhood
    H( ind, (yn-1)*xdim + xn )=1;

  end
end


%Normalize to that row norm = 1
H=H./(sqrt(sum(H.^2))'*ones(1,xdim*ydim));

