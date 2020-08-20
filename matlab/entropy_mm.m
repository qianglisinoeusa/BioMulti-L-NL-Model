function H = entropy_mm(p)

% mle estimator with miller-maddow correction

c = 0.5 * (sum(p>0)-1)/sum(p);  % miller maddow correction
p = p/sum(p);                   % empirical estimate of the distribution
idx = p~=0;
H = -sum(p(idx).*log2(p(idx))) + c;     % plug-in estimator of the entropy with correction