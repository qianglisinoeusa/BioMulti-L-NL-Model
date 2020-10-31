function [scanpath_out] = normalize_scanpath_rate(scanpath, samplerate)


	if size(scanpath,1)<1
		scanpath_out=[];
		return;
	end
	
	tstart=scanpath(1,3);
	tend=scanpath(end,3);

	timeintervals=tstart:samplerate:tend;
	totaltime=tend-tstart;

	scanpath_out(:,1)=interp1(scanpath(:,3),scanpath(:,1),timeintervals,'linear');
	scanpath_out(:,2)=interp1(scanpath(:,3),scanpath(:,2),timeintervals,'linear');
	scanpath_out(:,3)=timeintervals;
	scanpath_out(:,4)=timeintervals+samplerate;
	
    scanpath_out(find(scanpath_out(:,1)<min(scanpath(:,1))),:)=[];
    scanpath_out(find(scanpath_out(:,2)<min(scanpath(:,2))),:)=[];
    scanpath_out(find(scanpath_out(:,1)>max(scanpath(:,1))),:)=[];
    scanpath_out(find(scanpath_out(:,2)>max(scanpath(:,2))),:)=[];
    scanpath_out(find(isnan(scanpath_out(:,1))),:)=[];
    scanpath_out(find(isnan(scanpath_out(:,2))),:)=[];
    
end

