function label(im, str)
text(size(im, 2)/2, -40, str,...
    'Interpreter', 'none', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
% text(size(im, 2)/2, size(im, 1)+12, str,...
%     'Interpreter', 'none', 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
return
end
