function saliencyMap = saliency(img)
%
% Antonio Torralba 
% saliencyMap = saliency(img);
%
% If you call it without output arguments it will show two images.
% saliency = Product_i (hist(featuremap_i))
%
% This is the saliency map used in the Torralba 2006 paper 
% Torralba, Antonio and Oliva, Aude and Castelhano, Monica S. and
% Henderson, John M., (2006) Contextual guidance of eye movements and
% attention in real-world scenes: the role of global features in object 
% search. Psychological review.
% http://people.csail.mit.edu/torralba/GlobalFeaturesAndAttention/
%
% It is the basic saliency model without the task scene priors.


edges = 'reflect1';
pyrFilters = 'sp3Filters';

[nrows, ncols, cc]=size(img);
Nsc = 4;%maxPyrHt([nrows ncols], [15 15])-1; % Number of scales

[pyr, ind] = buildSpyr(double(mean(img,3)), Nsc, pyrFilters, edges);

weight = 2.^(1:Nsc);
sal = zeros(size(pyr));
saliencyMap = 1;
for b=2:size(ind,1)-1
   out =  pyrBand(pyr,ind,b);
   scale = floor((b-2)/6);

   [h,x] = hist(out(:), 100);
   h = h+1; h = h/sum(h);

   probOut = interp1(x, 1./h, out, 'nearest', min(h));   
   probOutR = imresize(probOut, [nrows ncols], 'bilinear');
   saliencyMap = saliencyMap + log(probOutR)*weight(scale+1);
end


if nargout == 0
   th = sort(saliencyMap(:));

   p = [0 .5 1]
   for n = 1:length(p)-1
       th1 = th(1+round(nrows*ncols*p(n)));
       th2 = th(round(nrows*ncols*p(n+1)));

       subplot(1,2,n)
       imagesc(img.*uint8(repmat((saliencyMap>th1).*(saliencyMap<=th2),[1 1 cc])))
       axis('equal'); axis('tight')
   end
end
