function  r_x4y=Adjancy4CorrPixel( P )

x1 = double(P(:,1:end-3));
y1 = double(P(:,4:end));
randIndex1 = randperm(numel(x1));
randIndex1 = randIndex1(1:5000);
x = x1(randIndex1);
y = y1(randIndex1);
r_x4y = corrcoef(x,y);
figure,
scatter(x,y);
xlabel('I(x,y)')
ylabel('I(x+4,y)')
title(['R=', num2str(r_x4y(1,2))])
end