%plot subroutine which 1) gives large fonts to ticks 2) has thicker lines
function plot_withbigfont(varargin)

clf
h1=axes;
h2=plot(varargin{:});
set(h1,'FontSize',25);
set(h2,'LineWidth',2);