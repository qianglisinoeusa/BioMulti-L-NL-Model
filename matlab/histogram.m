%subroutine to compute, plot, and save histograms
function histogram(data,xrange,filename)

global figurepath

clf
h=axes;
hist(data,xrange);
set(h,'FontSize',25)

print('-deps',[figurepath,filename,'.eps']);



