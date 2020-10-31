function [ ] = dmaps_converter( fdmap_path )

[path,name,extension]=fileparts(fdmap_path);
filename=name(7:end);
load(fdmap_path);
snames=fieldnames(fdMaps);
snames{end}=[]; snames=snames(~cellfun('isempty',snames));
n_subjects=numel(snames);

mkdir('pp');
for pp=1:n_subjects
    idx=str2num(snames{pp}(end-1:end));
    mkdir(['pp' '/' int2str(idx)]);
    dmap=normalize_minmax(getfield(fdMaps,snames{pp}));
    imwrite(dmap,['pp' '/' int2str(idx) '/' filename '.png']);
end

dmap=normalize_minmax(getfield(fdMaps,'all'));
imwrite(dmap,[filename '.png']);
end

