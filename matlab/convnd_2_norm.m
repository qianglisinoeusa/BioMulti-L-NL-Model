function y=convnd_2_norm(C1,W)


[a b c d]=size(W);

 y=zeros(size(C1,1)-size(W,1)+1,size(C1,2)-size(W,2)+1);
 normpatch=zeros(size(C1,1)-size(W,1)+1,size(C1,2)-size(W,2)+1);


%y=zeros(size(C1,1),size(C1,2));
%normpatch=zeros(size(C1,1),size(C1,2));

for (i=1:c)
    for(j=1:d)
     
      normpatch=normpatch+conv2(ones(a,1),ones(b,1),C1(:,:,i,j).^2,'valid'); % norm of each patch
     %sumfilt=ones(a,b);
     %sumfilt=sumfilt.*W(:,:,i,j);   % this just set to zeros the position of the sumfilter corresponding to the W
     %sumfilt(sumfilt~=0)=1;
     %normpatch=normpatch+conv2(C1(:,:,i,j).^2,sumfilt,'valid'); % norm of each patch
 

      % normpatch=normpatch+cudaconv(C1(:,:,i,j).^2,ones(a,b)); % norm of each patch
 

      y=y+conv2(C1(:,:,i,j),W(:,:,i,j),'valid');
      %y=y+convnfft(C1(:,:,i,j),W(:,:,i,j),'valid');   

     %  y=y+cudaconv(C1(:,:,i,j),W(:,:,i,j));

 

    end
end
      % normpatch=normpatch(a/2:end-a/2,b/2:end-b/2);
     %  y=y(a/2:end-a/2,b/2:end-b/2);




y= abs(y./sqrt(normpatch));

y(isnan(y))=0;
y(isinf(y))=0;

end
