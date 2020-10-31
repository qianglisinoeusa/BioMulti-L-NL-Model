function [ smap,scanpath,smaps] = saliency_lemeur( input_image, input_path )

	
	imwrite(input_image,'imgSource/default.png');
	
    addpath(genpath('GBVS'));	
	prior_smap=saliency_gbvs(input_image,input_path);
    prior_smap=mat2gray(prior_smap);
    imwrite(prior_smap,'imgSaliency/default.png');
    
	nbScanpaths=10;
	nbFixations=10;
	timeToRecover=8;
	nbCandidates=5;
	nppd=22;
	sceneType='naturalScenes'; %'naturalScenes', 'webPages', 'faces', 'landscapes'
    
	
    for g=1:nbFixations
        [scanpaths{g}, smaps(:,:,g)] = generateScanpath('imgSource/default.png', 'imgSaliency/default.png', nbScanpaths, g, timeToRecover, nbCandidates, nppd, sceneType); 
        smaps(:,:,g)=mat2gray(smaps(:,:,g));
    end
    smap=smaps(:,:,end);
    scanpath=scanpaths{nbFixations}; %last that has all fixations
    scanpath=edit_scanpath(scanpath,nbFixations);
    
    %we add an initial point at center of image (will not be used with metrics, but useful to make metrics correspond with point)
    centerFix=[round(size(input_image,2)/2),round(size(input_image,1)/2)];
    scanpath=[centerFix;scanpath];
end



function [escanpath] = edit_scanpath(scanpath,nbFixations)
    if nargin<2, nbFixations=numel(scanpath.posX); end
    
    escanpath=zeros(nbFixations,2);
    for g=1:nbFixations
%         %pick first scanpath (given several scanpath)
%         if numel(scanpath)==1
               escanpath(g,1)= scanpath(1).posX(g);
               escanpath(g,2)= scanpath(1).posY(g);
%         else 
%         %pick mean for all scanpath
%             all_posX=[];
%             all_posY=[];
%             for p=1:numel(scanpath)
%                all_posX=[all_posX, scanpath(p).posX(g)];
%                all_posY=[all_posY, scanpath(p).posY(g)];
%             end
%             escanpath(g,1)=round(mean(all_posX));
%             escanpath(g,2)=round(mean(all_posY));
%         end
    end

end