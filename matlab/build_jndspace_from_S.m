function [Y jnd] = build_jndspace_from_S(l,S)

L = 10.^l;
dL = zeros(size(L));

for k=1:length(L)
    thr = L(k)/S(k);

    % Different than in the paper because integration is done in the log
    % domain - requires substitution with a Jacobian determinant
    dL(k) = 1/thr * L(k) * log(10);
end

Y = l;
jnd = cumtrapz( l, dL );

end