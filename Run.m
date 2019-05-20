% Clear everything out of memory
clc;
clear all;

% Prepare image and checkpoint paths
addpath src -begin;
addpath src/cnn -begin;
addpath src/img -begin;

opts = appParamsNuclei();

% Prepare the data
if ~exist(fullfile([opts.expDir ' 1'],'imdb.mat'),'file')
    imdb = opts.prep.formatFunc();
    mkdir([opts.expDir ' ' num2str(1)]);
    save(fullfile([opts.expDir ' 1'],'options.mat'),'opts');
else
    load(fullfile([opts.expDir ' 1'],'imdb.mat'));
end
imdb.images.count = imdb.images.count./100;

%% Setup MatConvNet
run(fullfile('.','matconvnet','matlab','vl_setupnn'));
addpath(fullfile('.','matconvnet','examples'));

try
    if opts.useGpu
        I = gpuArray(single(1));
    else
        I = single(1);
    end
    vl_nnconv(I,I,[]);
    disp(['MatConvNet is compiled.'])
catch
    disp(['MatConvNet is not compiled, trying to compile now...']);
    vl_compilenn('enableGpu', true, ...
                 'cudaRoot', 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0', ...
                 'cudaMethod', 'nvcc', ...
                 'enableCudnn', false, ...
                 'cudnnRoot', 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0',...
                 'EnableImreadJpeg',false) ;
end

%% Train and evaluate the CNN

for i = 1:opts.network.numNetworks
    opts.train.randomSeed = opts.train.randomSeed + 1;
    
    if ~exist([opts.expDir ' ' num2str(i)],'dir')
        mkdir([opts.expDir ' ' num2str(i)]);
    end
    if ~exist(fullfile([opts.expDir ' ' num2str(i)],'imdb.mat'))
        save(fullfile([opts.expDir ' ' num2str(i)],'imdb.mat'),'imdb','-v7.3')
    end
    if ~exist(fullfile([opts.expDir ' ' num2str(i)],'options.mat'))
        save(fullfile([opts.expDir ' ' num2str(i)],'options.mat'),'opts')
    end
    opts.train.expDir = [opts.expDir ' ' num2str(i)];
    
    % Start from scratch
    net = initializeCNN(opts);
    
    % Add a subnetwork to predict cell counts
    filterSize = [opts.network.filterSize,opts.network.filterSize,...
                  512,512];
    filtSizes = 1:2:5;
    reductionLayers = (filterSize(4)/2)./2.^((filtSizes-1)/2);
    reductionLayers(end+1) = reductionLayers(end);
    depth = reductionLayers;
    [net,outLayer] = createInception(net,'U5_R1_bn','count_dense1',...
                              filterSize(3),filtSizes,reductionLayers,depth,...
                              opts.network.reluMethod,opts.network.leakyReLU);
    [net,outLayer] = createInception(net,outLayer,'count_dense2',...
                              filterSize(3),filtSizes,reductionLayers,depth,...
                              opts.network.reluMethod,opts.network.leakyReLU);
    avgPool = dagnn.Pooling('method', 'avg', 'poolSize', 7,'pad', [3 3], 'stride', 1);
    net.addLayer('avg_count',avgPool,...
                 {outLayer},{'avg_count'});
    previousLayer = 'avg_count';
    convSize = opts.prep.imgSize(1)/2^(opts.network.networkDepth-1);
    pad = 0;
    filterSize = [convSize convSize filterSize(4) 1];
    denseName = 'pred_count';
    dense = dagnn.Conv('size',filterSize,'pad',pad,'stride',1,'hasBias',true);
    net.addLayer('pred_count',dense,{previousLayer},{denseName},{[denseName '_f'] [denseName '_b']});
    net = init_params(net);
    previousLayer = denseName;
    l = dagnn.L2Loss();
    net.addLayer('L2', l, {'pred_count','count','count_weight'}, 'L2');
    opts.train.derOutputs = {opts.network.lossTypes(opts.network.loss),1,'L2',1};
    l = dagnn.L2Loss();
    l.type = 'RMSE';
    net.addLayer('RMSE', l, {'pred_count','count','count_weight'}, 'RMSE');

    % Call training function in MatConvNet
    opts.train.numEpochs = 15;
    [net, ~] = cnn_train_dag(net, imdb, opts.prep.getBatch, opts.train) ;
    
    opts.train.numEpochs = 30;
    opts.prep.mixWeightsWithLabels = false;
    opts.prep.getBatch = @(imdb,batch) get_batch(imdb,batch,opts);
    [net, ~] = cnn_train_dag(net, imdb, opts.prep.getBatch, opts.train) ;
    
    opts.train.numEpochs = 45;
    imdb.images.weight = sqrt(imdb.images.weight);
    opts.prep.getBatch = @(imdb,batch) get_batch(imdb,batch,opts);
    [net, ~] = cnn_train_dag(net, imdb, opts.prep.getBatch, opts.train) ;
    
    opts.train.numEpochs = 60;
    imdb.images.weight = sqrt(imdb.images.weight);
    opts.prep.getBatch = @(imdb,batch) get_batch(imdb,batch,opts);
    [net, ~] = cnn_train_dag(net, imdb, opts.prep.getBatch, opts.train) ;
    
    opts.train.numEpochs = 100;
    imdb.images.weight = imdb.images.weight.^4;
    opts.prep.getBatch = @(imdb,batch) get_batch(imdb,batch,opts);
    [net, info] = cnn_train_dag(net, imdb, opts.prep.getBatch, opts.train) ;
    
    opts.train.numEpochs = 160;
    imdb.images.weight = sqrt(imdb.images.weight);
    opts.prep.getBatch = @(imdb,batch) get_batch(imdb,batch,opts);
    [net, info] = cnn_train_dag(net, imdb, opts.prep.getBatch, opts.train) ;
    
    opts.train.numEpochs = 240;
    opts.prep.mixWeightsWithLabels = true;
    opts.prep.getBatch = @(imdb,batch) get_batch(imdb,batch,opts);
    [net, info] = cnn_train_dag(net, imdb, opts.prep.getBatch, opts.train) ;
    
    opts.train.numEpochs = 289;
    imdb.images.weight = (imdb.images.weight).^2;
    opts.prep.getBatch = @(imdb,batch) get_batch(imdb,batch,opts);
    [net, info] = cnn_train_dag(net, imdb, opts.prep.getBatch, opts.train) ;
    
    % Find the best network for nuclei segmentation
    load(['../Checkpoints 1/net-epoch-' num2str(opts.train.numEpochs) '.mat']);
    [best,ind] = max([stats.val.F1]);
    disp(['Best training F1 was ' num2str(best) ' at epoch ' num2str(ind) '. Saving...']);
    load(['../Checkpoints 1/net-epoch-' num2str(ind) '.mat']);
    save(['../Checkpoints 1/bestSeg.mat'],'net');
    
    % Find the best network for counting nuclei
    load(['../Checkpoints 1/net-epoch-' num2str(opts.train.numEpochs) '.mat']);
    [best,ind] = min([stats.val.RMSE]);
    disp(['Best training RMSE was ' num2str(best) ' at epoch ' num2str(ind) '. Saving...']);
    load(['../Checkpoints 1/net-epoch-' num2str(ind) '.mat']);
    save(['../Checkpoints 1/bestCount.mat'],'net');
    
    % Load the best network for nuclei counting, delete segmentation
    % components, and retrain for finetuning
    net = dagnn.DagNN.loadobj(net);
    net.removeLayer({net.layers(37:84).name});
    for i = 37:numel(length(layers))
        if isa(layers.block,'dagnn.ReLU')
            layers.block.leak = 0.05;
        end
    end
    opts.train.numEpochs = 100;
    opts.prep.getBatch = @(imdb,batch) get_batch_count(imdb,batch,opts);
    opts.train.expDir = [opts.expDir ' Count'];
    opts.train.derOutputs(1:2) = [];
    [net, info] = cnn_train_dag(net, imdb, opts.prep.getBatch, opts.train) ;
    
    % Load the best network for nuclei counting, delete segmentation
    % components, and retrain for finetuning
    load(['../Checkpoints 1/bestSeg.mat'],'net');
    opts = appParamsNuclei();
    net = dagnn.DagNN.loadobj(net);
    net.removeLayer({net.layers(85:end).name});
    opts.train.numEpochs = 400;
    opts.prep.getBatch = @(imdb,batch) get_batch_find(imdb,batch,opts);
    opts.train.expDir = [opts.expDir ' Find'];
    imdb.images.weight(imdb.images.label>0) = mean(imdb.images.weight(imdb.images.label>0));
    [net, info] = cnn_train_dag(net, imdb, opts.prep.getBatch, opts.train) ;
end