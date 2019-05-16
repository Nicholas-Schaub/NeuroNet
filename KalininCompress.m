path = '../Original Data/Kalinin/';
addpath('./src');

d = dir(path);

warning('off')
for i = 5:numel(d)
    ds = dir(fullfile(d(i).folder,d(i).name,'*.tif'));
    
    for j = 1:numel(ds)
        p = fullfile(ds(j).folder,ds(j).name);
        info = imfinfo(p);
        I = imread(p);
        for k = 1:numel(info)
            I = cat(3,I,imread(p,k,'info',info));
        end
        WriteTiff(I,p);
    end
    
    ds = dir(fullfile(d(i).folder,d(i).name,'masks_nuclei','*.tif'));
    
    for j = 1:numel(ds)
        p = fullfile(ds(j).folder,ds(j).name);
        info = imfinfo(p);
        I = imread(p);
        for k = 1:numel(info)
            I = cat(3,I,imread(p,k,'info',info));
        end
        WriteTiff(I,p);
    end
    
    ds = dir(fullfile(d(i).folder,d(i).name,'masks_nucleoli','*.tif'));
    
    for j = 1:numel(ds)
        p = fullfile(ds(j).folder,ds(j).name);
        info = imfinfo(p);
        I = imread(p);
        for k = 1:numel(info)
            I = cat(3,I,imread(p,k,'info',info));
        end
        WriteTiff(I,p);
    end
end
warning('on')