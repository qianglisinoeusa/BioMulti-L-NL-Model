function [foldernames labels file_paths ] =get_file_paths(directo)



% this function goes through the subfolders in alphabetical order and outputs
% the name of each subfolder and the path to all files in the subfolder


d=dir(directo);
names=lower({d.name});  % put names in lowercase
[a,b] = sort(names);    % sort in alphabetical order
d = d(b);
ct=0;
ctt=0;
foldernames=[];
file_paths=[];
labels=[];
for (k=3:length(d))
    ct=ct+1;
foldernames{ct}=d(k).name;

subdirecto=fullfile(directo,d(k).name);
dd=dir(subdirecto);
for (kk=3:length(dd))
    ctt=ctt+1;
file_paths{ctt}=fullfile(subdirecto,dd(kk).name);
labels(ctt)=ct;
end
end

end

