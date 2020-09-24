function  TR_xy_updated=TAdjancyCorrPixel( P )

TR_xy=[];
TR_xy_updated=[];
for i=0:40
	x1 = double(P(:,1:end-i));
	y1 = double(P(:,i+1:end));
	randIndex1 = randperm(numel(x1));
	randIndex1 = randIndex1(1:5000);
	x = x1(randIndex1);
	y = y1(randIndex1);
	Tr_xy = corrcoef(x,y);
	TR_xy=[TR_xy; {Tr_xy}];
end

for j=1:40
	TR_xy_updated=[TR_xy_updated, {TR_xy{j,1}(1,2)}];
end
end
