% DCT Signature (DCTS)
%   [3] X. Hou, J. Harel, and C. Koch, "Image Signature: Highlighting
%       sparse salient regions," in PAMI, 2011.
%Params fixed by authors
  
function outMap = DCTS(imagename,fr)
if nargin < 2
    fr=1;
end

%redimensiona a dimension maior a 64
param = default_pami_param();
img=imread(imagename);
img = imresize(img, param.mapWidth/size(img, 2));
sze=size(img);
cSalMap = zeros(size(img));

if (numel(sze)==3)
    oppoImg = applycform(img,makecform('srgb2lab'));
    
    for i = 1:3
        cSalMap(:,:,i) = idct2(sign(dct2(oppoImg(:,:,i)))).^2;
    end
    
    outMap = sum(cSalMap, 3);
    
else
    outMap = idct2(sign(dct2(img(:,:)))).^2;
    
end

outMap = mynorm(outMap, param);
outMap = filter2(fspecial('gaussian',param.maxhw,param.sig),outMap);

end

