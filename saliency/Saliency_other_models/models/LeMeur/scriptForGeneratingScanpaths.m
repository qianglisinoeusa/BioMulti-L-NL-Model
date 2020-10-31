function [dataScanPath,saliency] = scriptForGeneratingScanpaths(directoryOrigPict, directoryOrigSal, directoryOutput, nbScanpaths, nbFixations, timeToRecover, nbCandidates, nppd, sceneType)
close all;

%----------------------------------------------------------------------
%nbScanpaths   = 10;
%nbFixations   = 10;
%timeToRecover = 8 ; %8; -1 => WTA there is no recovery
%----------------------------------------------------------------------
imglist = dir([directoryOrigPict '*.jpg']); 
fnum    = length(imglist);

% fnum = 1; % 1 for the model which is blind to low-level visual features.
%----------------------------------------------------------------------
for iImage=1:fnum    % 1 for the model which is blind to low-level visual features.
    fprintf('\n Processing file ........... %s', imglist(iImage).name);
    fileNameImageOrig = [directoryOrigPict imglist(iImage).name]
    fileNameImageSal = [directoryOrigSal imglist(iImage).name]
    fileNameImageSal = strrep(fileNameImageSal, '.jpg', '.pgm');
         
    [dataScanPath, saliency] = generateScanpath(fileNameImageOrig, fileNameImageSal, nbScanpaths, nbFixations, timeToRecover, nbCandidates, nppd, sceneType) ;
    
    str = [directoryOutput imglist(iImage).name];
         
    strSal = strrep(str, '.jpg', '.pgm');
    saliencyNorm = normalizeData(saliency);
    imwrite(saliencyNorm, strSal);
    
    strStat = strrep(str, '.jpg', '.stat');
    fd=fopen(strStat, 'w+');    
    
    for i=1:nbScanpaths
        for j=1:nbFixations
            fprintf(fd, '%d %d 100 ', dataScanPath(i).posX(j), dataScanPath(i).posY(j));
        end
        fprintf(fd, '-1 -1 -1\n');
    end
    fprintf(fd, '-1 -1 -1');
    fclose(fd);
end
end

function imNorm=normalizeData(im)
maxi = max(max(im));
imNorm = im./maxi;
end
