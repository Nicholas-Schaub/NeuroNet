networkPath = fullfile('.','models','bestFind.mat');
optionsPath = fullfile('..','Checkpoints 1','options.mat');
addpath(genpath('src'));

run(fullfile('.','matconvnet','matlab','vl_setupnn'));
addpath(fullfile('.','matconvnet','examples'));

load(networkPath);
load(optionsPath);

%% Setup MatConvNet
run(fullfile('.','matconvnet','matlab','vl_setupnn'));
addpath(fullfile('.','matconvnet','examples'));

%% Load model and modify
nn = dagnn.DagNN.loadobj(net);
nn.mode = 'test';
nn.removeLayer({nn.layers(85:end).name});
dag = mcn2mat(nn,opts);

exportONNXNetwork(dag,'nuclei_count.onnx','OpsetVersion',9)

% %% Load a demo image
% I = imread('../Original Data/Coelho/data/images/dna-images/gnf/dna-8.png');
% 
% %% Layer by layer comparison
% for i = 1:numel(nn.vars)
%     nn.vars(i).precious = 1;
% end
% 
% %% Test the network
% S_mcn = CNNAttribute(I(:,:,1),nn,opts,20,false);
% S_mat = CNNAttribute(I(:,:,1),dag,opts,20,false);
% figure(1);
% subplot(1,3,1), imagesc(S_mcn);
% subplot(1,3,2), imagesc(S_mat);
% subplot(1,3,3), imagesc((S_mcn - S_mat).^2);
% 
% %% Compare layerwise errors
% J = Preprocess(single(I(:,:,1)),opts);
% J = Tile(J,opts);
% J = J(:,:,:,1:10);
% nn.eval({'input',J,'label',zeros(opts.prep.lblSize(1),opts.prep.lblSize(2),size(J,3),size(J,4),'single')});
% 
% sse = zeros(1,numel(net.vars));
% for i = 1:numel(nn.vars)
%     try
%         K = activations(dag,J,nn.vars(i).name);
%     catch err
%         warning(['Could not find layer in dag network: ' nn.vars(i).name]);
%         continue;
%     end
%     sse(i) = sqrt(mean((K(:) - nn.vars(i).value(:)).^2));
% end
% 
% figure, plot(sse,'ro');
% xticks(1:numel(sse));
% xticklabels({net.vars.name});
% xtickangle(45);