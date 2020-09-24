function  r_x2y=Adjancy2CorrPixel( P )

x1 = double(P(:,1:end-2));
y1 = double(P(:,3:end));
randIndex1 = randperm(numel(x1));
randIndex1 = randIndex1(1:5000);
x = x1(randIndex1);
y = y1(randIndex1);
r_x2y = corrcoef(x,y);
figure,
scatter(x,y);
xlabel('I(x,y)')
ylabel('I(x+2,y)')
title(['R=', num2str(r_x2y(1,2))])
end
