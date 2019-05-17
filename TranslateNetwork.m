networkPath = fullfile('..','Checkpoints 1','bestSeg.mat');
optionsPath = fullfile('..','Checkpoints 1','options.mat');
addpath(genpath('src'));

load(networkPath);
load(optionsPath);

%% Setup MatConvNet
run(fullfile('.','matconvnet','matlab','vl_setupnn'));
addpath(fullfile('.','matconvnet','examples'));

%% Load model and modify
nn = dagnn.DagNN.loadobj(net);
nn.removeLayer({nn.layers(85:end).name});

%% Translate to matlab network
dag = mcn2mat(nn,opts);

%% Determine similarity of networks
