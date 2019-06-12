%% Clear workspace, set up file paths, and check for bioformats
clear;
clc;

% output path for volumetric data
out_vol = fullfile('.','data','volumes');

% output path for label data
out_lbl = fullfile('.','data','labels');

addpath(genpath('src'));
addpath(genpath('bfmatlab'));

[status,version] = bfCheckJavaPath();
disp(['Loaded Bio-Formats version ' version]);
bfUpgradeCheck(false);

%% Load mrc and rec files, save as compressed tiffs
folders = dir(fullfile('..','NIAID_Bryan'));

mkdir(out_vol);

images = [];
for folder = 1:numel(folders)
    images = [images; dir(fullfile(folders(folder).folder,folders(folder).name,'*.rec'))];
    images = [images; dir(fullfile(folders(folder).folder,folders(folder).name,'*.mrc'))];
    images = [images; dir(fullfile(folders(folder).folder,folders(folder).name,'*.hdf'))];
end
for image = 1:numel(images)
    disp(['Processing file: ' images(image).name])
    f = fullfile(images(image).folder,images(image).name);
    [pathstr, name, ext] = fileparts(f);

    out_file = fullfile(out_vol,[name '.ome.tif']);
    
    if exist(out_file,'file')
        warning(['The following file already exists and will not be resaved: ' out_file])
        continue;
    end
    
    if ~strcmp(ext,'.hdf')
        data = bfopen(f);
        pixels = numel(data{1,1}{1,1})*numel(data{1,1});
    else
        continue;
%         data = h5read(f,'/MDF/images/0/image');
%         pixels = numel(data);
    end
    if pixels > 2^32-1
        bigTiff = true;
    else
        bigTiff = false;
    end
    D = cat(3,data{1,1}{:,1});
    bfsave(D,out_file,...
          'Compression','zlib',...
          'BigTiff',bigTiff,...
          'metadata',data{1,4});
    clear data;
end

%% Load and convert rec files
mkdir(out_lbl);

images = [];
for folder = 1:numel(folders)
    images = [images; dir(fullfile(folders(folder).folder,folders(folder).name,'*.am'))];
    images = [images; dir(fullfile(folders(folder).folder,folders(folder).name,'*.tif'))];
end
for image = 1:numel(images)
    disp(['Processing file: ' images(image).name])
    f = fullfile(images(image).folder,images(image).name);
    [pathstr, name, ext] = fileparts(f);

    out_file = fullfile(out_lbl,[name '.ome.tif']);
    
    if exist(out_file,'file')
        warning(['The following file already exists and will not be resaved: ' out_file])
        continue;
    end
    
    data = bfopen(f);
    pixels = numel(data{1,1}{1,1})*numel(data{1,1});

    if pixels > 2^32-1
        bigTiff = true;
    else
        bigTiff = false;
    end
    bfsave(cat(3,data{1,1}{:,1}),out_file,...
          'Compression','zlib',...
          'BigTiff',bigTiff,...
          'metadata',data{1,4});
    clear data;
end

%% Load and convert amira files

clear;
addpath(genpath('src'));

out_vol = fullfile('.','data','labels');

folders = dir(fullfile('..','NIAID_Bryan'));

images = [];
for folder = 1:numel(folders)
    images = [images; dir(fullfile(folders(folder).folder,folders(folder).name,'*.am'))];
end

mkdir(out_vol);

for i = 2:numel(images)
    disp(['Processing file: ' images(i).name]);
    
    % Get mrc file info
    [h s]=LoadData_Amira(fullfile(images(i).folder,images(i).name));
    
    s = reshape(s,size(s,1),size(s,2),[],size(s,3));
    
    if isfield(h,'Materials')
        tags.ImageDescription = '';
        for j = 1:numel(h.Materials)
            tags.ImageDescription = [tags.ImageDescription num2str(j) ',' h.Materials{j}.Name ','];
        end
        tags.ImageDescription = tags.ImageDescription(1:end-1);
    else
        tags = struct;
    end
    
    WriteTiff(s,fullfile(out_vol,[images(i).name(1:end-2) 'tif']),tags);
    
    clear s;
end