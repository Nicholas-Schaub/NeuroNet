function [opts] = appParamsNuclei()

opts = struct;

% Paths to directories
opts.expDir = '../Checkpoints';
opts.imgPath = '../Original Data';
opts.lblPath = '../Checkpoints';

% Network Structure
% Note: The general network structure is a residual network regardless of
% what other types of settings are used.
% TODO: only works for PixAttrib at the moment. Need to implement other
% network types for initializeCNN and format_data
opts.network.typeName = containers.Map([1:6],...
                                      {'ImgClass';
                                       'ImgReg';
                                       'ImgAttrib';
                                       'PixClass';
                                       'PixReg';
                                       'PixAttrib'});
opts.network.type = 6;
opts.network.complexity = 5;
opts.network.networkDepth = 5;
opts.network.layerDepth = 1;
opts.network.numClass = 1; %TODO finish implementing in initializeCNN and format_data
opts.network.filterSize = 3;
opts.network.layerTypeName = containers.Map([1:3],...
                                            {'normal',...
                                             'reduced',...
                                             'inception'});
opts.network.layerType = 2;
opts.network.leakyReLU = 0.5;
opts.network.reluMethod = 'elu'; % Must be elu or relu
opts.network.useNIN = false;
opts.network.useBatchNorm = true;
opts.network.useDropout = 0;
opts.network.numNetworks = 1;
opts.network.lossTypes = containers.Map([1:3],...
                                        {'L2';          % regression
                                         'log';         % classification
                                         'logistic'});  % attributes
opts.network.loss = 3;
opts.network.metricTypes = containers.Map([1:5],...
                                          {'classerror';  % classification
                                           'top5error';   % classification
                                           'binaryerror'; % attributes
                                           'F1';          % attributes
                                           'F2'});        % attributes
opts.network.metrics = [3 4];

% GPU Usage
opts.useGpu = true;
opts.verbose = false;
opts.storeGpu = false; % store data on GPU to minimize data transfer

% Training options
opts.train = struct;
opts.train.batchSize = 20;
opts.train.numSubBatches = 1;
opts.train.numEpochs = 1000;
opts.train.continue = true;
opts.train.gpus = [1];
opts.train.learningRate = 0.0001;
opts.train.weightDecay = 0.0005;
opts.train.expDir = opts.expDir;
opts.train.solver = @solver.adam;
opts.train.solverOpts = struct('beta1', 0.9, 'beta2', 0.999, 'eps', 1e-8) ;
opts.train.saveSolverState = true;
opts.train.randomSeed = 0;
opts.train.plotStatistics = true;
% DO NOT EDIT
opts.train.derOutputs = {opts.network.lossTypes(opts.network.loss),1};

% Jitter options
opts.jitter = struct;
opts.jitter.randomCrop = true; % TODO: Randomly crop during training
opts.jitter.flipHorizontal = true; % Randomly flip horizontally during training
opts.jitter.flipVertical = true; % Randomly flip vertically during training

% Image loading and preprocessing options
opts.prep = struct;
opts.prep.channels = [1]; % Specify color channels to be used
opts.prep.imgSize = [256 256]; % Dimensions must be a power of 2
opts.prep.lblSize = [200 200];
opts.prep.holdout = 0.3; % Proportion of images used for validation during training
opts.prep.preCrop = false; % Should images be cropped prior to training
% thresholding - set to empty for no thresholding. Otherwise, select a
%                value for binary thresholding
%              - for classification, set to bin edges just like the
%                histcounts function. There should be C+1 values, so if you
%                have 3 classes there should be 4 values.
opts.prep.binThreshold = 0.5; % Set to empty for no thresholding
opts.prep.foreVal = 1;
opts.prep.backVal = -1;
opts.prep.foreWeightType = 'distance'; % Can be distance or intensity
opts.prep.backWeightType = 'intensity'; % Can be distance or intensity
opts.prep.foreWeightThresh = 10; % Threshold the foreground weights
opts.prep.backWeightThresh = -10;
opts.prep.foreWeightAmp = -0.5;
opts.prep.backWeightAmp = 2;
opts.prep.foreWeightThreshDir = 1; % 1 for positive threshold, -1 for negative threshold
opts.prep.backWeightThreshDir = 1;
opts.prep.windowSize = 255; % Value must be odd
opts.prep.normImage = true;
opts.prep.normWeights = false;
opts.prep.maxNorm = 6;
opts.prep.maxWeight = 10;
opts.prep.mixWeightsWithLabels = true;
opts.prep.preLoad = false;
opts.prep.getBatch = @(imdb,batch) get_batch(imdb,batch,opts);
opts.prep.formatFunc = @() format_data_nuclei(opts);

end