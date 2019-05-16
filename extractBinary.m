% Clear everything out of memory
clc;
clear all;

% Prepare image and checkpoint paths
addpath src -begin;
addpath src/cnn -begin;
addpath src/img -begin;
run(fullfile('.','matconvnet','matlab','vl_setupnn'));
addpath(fullfile('.','matconvnet','examples'));

opts = appParamsNuclei();

% Prepare the data
load(fullfile([opts.expDir ' 1'],'imdb.mat'));
imdb.images.data = [];
imdb.images.weight = [];
imdb.images.label = [];

%%
load(['../Checkpoints 1/bestSeg.mat']);

net = dagnn.DagNN.loadobj(net);

images = unique(imdb.images.imgFile);
labels = unique(imdb.images.lblFile);

h = waitbar(0,'Processing images...');
for b = 1:numel(images)
    I = imread(images{b});
    B = imread(labels{b});
    C = CNNsegment(I,net,opts,20,false);
    C = imfill(C,'holes');
    C = imclose(C,ones(3));
    C = imopen(C,ones(3));
    
    out_path = strrep(images{b},'..','../Parsed Data');
    o = strsplit(out_path,'/');
    mkdir([fullfile(o{1:end-1}) '/']);
    WriteTiff(I,out_path);
    
    out_path = strrep(labels{b},'..','../Parsed Data');
    o = strsplit(out_path,'/');
    mkdir([fullfile(o{1:end-1}) '/']);
    WriteTiff(B,out_path);
    
    out_path = strrep(out_path,'labels','neuronet');
    o = strsplit(out_path,'/');
    mkdir([fullfile(o{1:end-1}) '/']);
    WriteTiff(uint8(C),out_path);
    
    waitbar(b/numel(images),h);
end