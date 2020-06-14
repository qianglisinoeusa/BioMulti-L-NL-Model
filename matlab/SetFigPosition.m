function SetFigPosition(Setting)
% get current figure
% get(gcf, 'Position')

if Setting == 1     % main screen, upper right side
    Pos = [795 359 798 457];
elseif Setting == 2 % main screen, total right side
    Pos = [809 39 784 777];
elseif Setting == 3 % second screen, large size
    Pos = [1879 -155 1631 971];
elseif Setting == 4 % second screen, middle size
    figure(1)
    Pos = [2651 214 859 602];
elseif Setting == 5 % main screen, large size, small height
    figure(1)
    Pos = [10 318 1583 498];
elseif Setting == 6 % second screen, large size, small height
    figure(1)
    Pos = [1666 266 1729 464];
else
    error('This setting is not defined')
end
