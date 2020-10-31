function [scanpath] = limit_scanpath(scanpath,limits)
%     size(scanpath)
%     for row=1:size(scanpath,1)
%             if scanpath(row,1) < 0 || scanpath(row,2) < 0 || scanpath(row,1) > limits(2) || scanpath(row,2) > limits(1)
%                 scanpath(row) = [];
%             end
%     end
if size(scanpath,1)>0
scanpath(scanpath(:,1) < 1,:) = [];
scanpath(scanpath(:,2) < 1,:) = [];
scanpath(scanpath(:,1) > limits(2),:) = [];
scanpath(scanpath(:,2) > limits(1),:) = [];
scanpath(isnan(scanpath(:,1)),:) = [];
scanpath(isnan(scanpath(:,2)),:) = [];
end

end

