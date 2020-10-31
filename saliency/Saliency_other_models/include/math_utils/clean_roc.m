function [ tp,fp ] = clean_roc( tp,fp )

%erase duplicate cases
%[tp,idx_unique]=uniquetol(tp,0.0001);
[tp,idx_unique]=uniquetol(tp,0.001);
fp=fp(idx_unique);

%erase zeros
idx_nonzeros=find(fp~=0);
tp=tp(idx_nonzeros);
fp=fp(idx_nonzeros);

%erase non progressive fp values
% i=2;
% li=length(fp);
% while i<li
%     if fp(i)<fp(i-1)
%        fp(i)=[];
%        tp(i)=[];
%        li=li-1;
%     else
%         i=i+1;
%     end
% end

fp=interp1(tp,fp,0:1/256:1);
tp=0:1/256:1;


end

