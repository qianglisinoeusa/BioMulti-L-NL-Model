function y = sign_pow( x, e )
y = sign(x) .* abs(x).^e;
end