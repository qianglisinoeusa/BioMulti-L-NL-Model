function [ path ] = addsuffix( path , suffix)

[folder,filename,extension]=fileparts(path);
path=[folder '/' filename suffix extension];


end

