function outMap = oppoSig2(img)

%redimensiona a dimension maior a 64
param = default_pami_param();
img = imresize(img, param.mapWidth/size(img, 2));

cSalMap = zeros(size(img));
oppoImg = applycform(img,makecform('srgb2lab'));

for i = 1:3
	cSalMap(:,:,i) = idct2(sign(dct2(oppoImg(:,:,i)))).^2;
end

outMap = sum(cSalMap, 3);
outMap = mynorm(outMap, param);
outMap = filter2(fspecial('gaussian',param.maxhw,param.sig),outMap);

end

