function  r_xy=AdjancyCorrPixel( P )

x1 = double(P(:,1:end-1));
y1 = double(P(:,2:end));
randIndex1 = randperm(numel(x1));
randIndex1 = randIndex1(1:5000);
x = x1(randIndex1);
y = y1(randIndex1);
r_xy = corrcoef(x,y);
figure,
scatter(x,y);
xlabel('I(x,y)')
ylabel('I(x+1,y)')
title(['R=', num2str(r_xy(1,2))])
end
