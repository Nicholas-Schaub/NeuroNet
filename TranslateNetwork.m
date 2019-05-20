networkPath = fullfile('.','models','bestFind.mat');
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

%% Load a demo image
I = imread('../Original Data/Coelho/data/images/dna-images/gnf/dna-8.png');

%% Run the network using MCN
S_mcn = CNNAttribute(I(:,:,1),nn,opts,20,false);
dag = mcn2mat(nn,opts);
S_mat = CNNAttribute(I(:,:,1),dag,opts,20,false);