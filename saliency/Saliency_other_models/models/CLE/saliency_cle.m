function [ smap,scanpath,smaps] = saliency_cle( input_image, input_path )
	

	imwrite(input_image,'img/default.png');
	%[smap]=cleComputeSaliency(input_image,'SPECTRAL_RES');

	configFileName = 'config_simple';
	nObs=5;
	nbFixations=10;
    
    %for CLE, the resulting smap is the same as xhou (Spectral Residual Saliency)
    
	for g=1:nbFixations
  	    [scanpaths{g}, smaps(:,:,g)] = cleGenerateScanpath(configFileName, nObs,g);
    end
    smap=smaps(:,:,end);
    scanpath=scanpaths{nbFixations};
    close all;
    
    
    scanpath=fliplr(scanpath(:,1:end)');
    
    %it has already initial point
%     %we add an initial point at center of image (will not be used with metrics, but useful to make metrics correspond with point)
%     centerFix=[round(size(input_image,2)/2),round(size(input_image,1)/2)];
%     scanpath=[centerFix,scanpath];
    
    smap=mat2gray(smap);
	
end
