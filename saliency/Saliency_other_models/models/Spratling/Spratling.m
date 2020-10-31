function [smap] = Spratling(I)

[imsizefac,crop,max_radius,sigma]=common_param_values;
w=[];
for s=sigma
  w=filter_definitions_V1_simple(s,w);
end

sg=1.5;
[X]=preprocess_V1_input(I,sg);
A=[];
iterations=10;
[y,r,e,ytrace,rtrace,etrace]=network_dim_conv(w,X,iterations,A);

R=max(cat(4,etrace{:}),[],4);

for t=1:iterations
  Rmaxiter=max(R(:,:,t));
end

Rmax=R(:,:,iterations);

smap=Rmax;

end
