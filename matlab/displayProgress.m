function displayProgress(i,nIterations,titleString)
% displayProgress(i,nIterations,titleString)
%-----------------------------------------------------------------------------------------
% DISPLAYPROGRESS - prints the progress of the iterations in any loop in
% command window. Only one instance of this function should be used. Do do
% not put this function in multiple places in the same function.
%
% INPUTS:
%   nIterations - maximum number of iterations in the loop.
%   i           - current iteration count.
%   titleString - optional string to display as title to the progress.
% EXAMPLE: 
%   displayProgress(ithLoop,120,'Objects processed')
%   displayProgress(ithLoop,120)
% This function is called by:
% This function calls:
% MAT-files required:
%
% See also: 
%
% Author: Mani Subramaniyan
% Date created: 2009-05-01
% Last revision: 2009-05-01
% Created in Matlab version: 7.5.0.342 (R2007b)
%-----------------------------------------------------------------------------------------

stride = (nIterations < 10)*100/nIterations + (nIterations >=10) * 10;
stepValue = (nIterations*stride/100);

persistent counterValue m

% Display counter:
if i==1
    m = stepValue; 
    counterValue = 0;
    if nargin > 2
        fprintf('%s\n',titleString)
    end
    fprintf('%s %%\n','  0');
end


if i >= m
     counterValue = counterValue + 1;
    if counterValue <=(100/stride)
        fprintf(' %0.0d %%\n',round(counterValue*stride));
    end
    m = stepValue * counterValue;
end

if i==nIterations
    clear('counterValue','m');
end