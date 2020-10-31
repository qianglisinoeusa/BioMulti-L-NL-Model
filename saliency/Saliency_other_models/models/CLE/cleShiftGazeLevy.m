function [x_new, y_new, candx, candy ] = cleShiftGazeLevy(alpha_stable,...
                                                        beta_stable, ...
                                                        gamma_stable, ...
                                                        delta_stable, n,...                                        
                                                        x_old, y_old, gazeDir, ...
                                                        k_P, k_R, dV_x, dV_y, h,...
                                                        rows, cols, sal)
                                                    
% cleShiftGazeLevy - Computes the gaze shift as a Levy walk step
%
% Synopsis
%          [x_new, y_new, candx, candy ] = cleShiftGazeLevy(alpha_stable,...
%                                                       beta_stable, ...
%                                                       gamma_stable, ...
%                                                       delta_stable, n,...                                        
%                                                       x_old, y_old, gazeDir, ...
%                                                       k_P, k_R, dV_x, dV_y, h,...
%                                                       rows, cols, sal)
%
% Description
%   Computes the gaze shift as a Levy walk step implemented trough a
%   Langevin-like Stochastic Differential Equation (SDE)[1], where the random
%   component is sampled from an alpha-stable distribution. The main
%   difference from [1] is that the choice of the next gaze location is
%   obtained through an internal simulation step: a number n of candidates gaze
%   shifts is preliminarly sampled and evaluated against a gain function [2]. 
%   The best among n candidate shift is eventually retained 
%
% Inputs ([]s are optional)
%   (double) alpha_stable        alpha parameter of the alpha-stable distribution 
%   (double) beta_stable         simmetry parameter of the alpha-stable distribution 
%   (double) gamma_stable        dispersion parameter of the alpha-stable distribution 
%   (double) delta_stable        location parameter of the alpha-stable distribution
%   (int)    n                   number of candidates to sample
%   (int)    x_old               x coord of the old gaze position
%   (int)    y_old               y coord of the old gaze position
%   (double) gazeDir             the old gaze direction
%   (double) k_P, k_R            weights for deterministic and static component of 
%                                the Langevin equation
%   (matrix) dV_x, dV_y          x, y component of grad(V)
%   (double) h                   the time step of the Langevin-like SDE
%   (int)    rows                number of rows
%   (int)    cols                number of columns
%   (matrix) sal                 the salience map
%
% Outputs ([]s are optional)
%   
%   (int)    x_old               x coord of the new gaze position
%   (int)    y_old               y coord of the new gaze position
%   (vector) candx               x coord of the candidate gaze positions
%   (vector) candy               y coord of the candidate gaze positions
%
%
% See also
%   stabrnd
%
% Requirements
%   Image Processing toolbox
%   Statistical toolbox

% References
%   [1] G. Boccignone and M. Ferraro, Modelling gaze shift as a constrained
%       random walk, Physica A, vol. 331, no. 1, pp. 207-218, 2004.
%   [2] G. Boccignone and M. Ferraro, Feed and fly control of visual 
%       scanpaths for foveation image processing, Annals of telecommunications - 
%       Annales des télécommunications 
%       2012 (in press).
%   
%
% Authors
%   Giuseppe Boccignone <Giuseppe.Boccignone(at)unimi.it>
%
% License
%   The program is free for non-commercial academic use. Please 
%   contact the authors if you are interested in using the software
%   for commercial purposes. The software must not modified or
%   re-distributed without prior permission of the authors.
%
% Changes
%   20/01/2011  First Edition
%   
    	

% Distances drawn from the stable random number generator
r = stabrnd(alpha_stable, beta_stable, gamma_stable, delta_stable, 1, n);



%----------------------------------------------------------------------
% Generate randomly a direction theta
%----------------------------------------------------------------------
%theta = 2*pi*rand(1,n); %old and simple solution

%----------------------------------------------------------------------
% Generate randomly a direction theta from a uniform distribution 
% between  -pi and pi and as a function of previous
% direction
%----------------------------------------------------------------------
theta = 2*pi*rand(1,n)-pi+ gazeDir;

%----------------------------------------------------------------------
% Compute  new gaze position of the FOA via Langevin equation
%----------------------------------------------------------------------

% two 1xn row vectors

x_new = round(x_old + h*(- k_P*dV_x(round(x_old))+k_R * r.*cos(theta))) ;
y_new = round(y_old + h*(- k_P*dV_y(round(y_old))+k_R * r.*sin(theta))) ;

% Verifies if the generated gaze shift is located within the image
validcord = find((x_new >0) & (x_new <rows) & (y_new >0) & (y_new <cols));

% if no valid coordinates have been sampled....
if numel(validcord)==0
    repeatLevyTry=true;
    % .... repeat sampling
    while repeatLevyTry            
        % Distances drawn from the stable random number generator
        r = stabrnd(alpha_stable, beta_stable, gamma_stable, delta_stable, 1, n);


        %theta = 2*pi*rand(1,n); 

        %----------------------------------------------------------------------
        % Generate randomly a direction theta from a uniform distribution 
        % between  -pi and pi and as a function of previous
        % direction
        %---------------------------------------------------------------------- 
        theta = 2*pi*rand(1,n)-pi+ gazeDir;

        %----------------------------------------------------------------------
        % Compute  new gaze position of the FOA via Langevin-like SDE equation
        %----------------------------------------------------------------------

        %two 1xn row vectors
        %k_P=0; k_R=1;
        x_new = round(x_old + h*(- k_P*dV_x(round(x_old))+k_R * r.*cos(theta))) ;
        y_new = round(y_old + h*(- k_P*dV_y(round(y_old))+k_R * r.*sin(theta))) ;

        % Verifies if the generated gaze shift is located within the image
        validcord =find((x_new >0) & (x_new <rows) & (y_new >0) & (y_new <cols));
        if numel(validcord)~=0
            % stops the sampling loop 
            repeatLevyTry = false;
        end           

    end

end
% Retains only the valid ones
x_new = x_new(validcord);
y_new = y_new(validcord);

candx=x_new; candy=y_new;



%----------------------------------------------------------------------
%
% Compute the function $\varphi(s)$ 
% this modifies the pure Levy flight, in that
% the probability $p({\vec r}_{new}\mid {\vec r})$ to move from a
% site $\vec r$ to the next site ${\vec r}_{new}$ depends on the
% "strength" of a bond $\varphi$ that exists between them
%     
%----------------------------------------------------------------------
varPhi = zeros(size(x_new)); %allocating 	
for ww=1:length(x_new)
    %varPhi(ww)= exp(-beta_P*sal(sal(x_old,y_old)- x_new(ww),y_new(ww)));
    varPhi(ww) = sal(x_new(ww),y_new(ww)) - sal(x_old,y_old);
end   


%best among N
idxMax=find( varPhi == max(max(varPhi)));

x_new = x_new(idxMax);
y_new = y_new(idxMax);

%return just one point if multiple max
x_new = x_new(1);
y_new = y_new(1);

end