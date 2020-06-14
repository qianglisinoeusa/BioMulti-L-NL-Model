%compute correlation coefficients of variables, 
%giving out vector that give the correlations
%in a single vector with no variances or repetitions
%(for plotting histograms etc.)

function corrvector=computecorrelations(X)

%compute correlation coefficients, using transpose of input vector
corrmatrix=corrcoef(X');
%this funny operation simply gives 
%all the element above the diagonal as a single vector:
[dummy,dummy,corrvector]=find(triu(corrmatrix)-diag(diag(corrmatrix)));

