function [opts] = AppParams()
%% appParams.m
% This function contains a set of parameters that can be used across all
% functions in this repository.
%
% There are four main attributes of the options structure returned by this
% function:
% opts - General information,file paths, gpu usage, and parallel processing
% opts.network - CNN network construction attributes
% opts.train - CNN training atrributes
% opts.prep - Image preprocessing attributes

opts = struct;

%% General options
% Paths to directories
opts.expDir = '.\Path\To\Checkpoints'; % will make the dir if it doesn't exist
opts.imgPath = '.\Path\To\Images';
opts.lblPath = '.\Path\To\Labels';
opts.useGpu = true;
opts.numGpus = [1]; % can be a scalar (n) that will find the first n gpus,
                    % or can be a vector of gpu indices
opts.cudaPath = []; % if empty, will try to determine the cuda install path
                    % and will compile cpu version if can't be found
opts.verbose = false; % not implemented, will provide debugging information
                      % in the command window when set to true
opts.storeGpu = false; % not implemented, will store data on GPU to
                       % minimize data transfer
opts.loadParallel = false; % when true, loads the next set of images while
                           % processing the current set.
opts.writeParallel = false; % when true, start a parallel process to write
                            % an image.

%% Network Structure
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
opts.network.type = 4;
opts.network.complexity = 5;
opts.network.networkDepth = 5;
opts.network.layerDepth = 1;
opts.network.numClass = 3; %TODO finish implementing in initializeCNN and format_data
opts.network.classLabels = {'gray',...
                            'nothing',...
                            'white'};
opts.network.filterSize = 5;
opts.network.layerTypeName = containers.Map([1:3],...
                                            {'normal',...
                                             'reduced',...
                                             'inception'});
opts.network.layerType = 2;
opts.network.reluMethod = 'elu'; % must be 'elu' or 'relu'
opts.network.leakyReLU = 1;
opts.network.useNIN = false;
opts.network.useBatchNorm = true;
opts.network.useDropout = 0.5; % 0-1 value, 0 or [] means no dropout
opts.network.numNetworks = 1;
opts.network.lossTypes = containers.Map([1:3],...
                                        {'L2';          % regression
                                         'log';         % classification
                                         'logistic'});  % attributes
opts.network.loss = 2;
opts.network.metricTypes = containers.Map([1:5],...
                                          {'classerror';  % classification
                                           'top5error';   % classification
                                           'binaryerror'; % attributes
                                           'F1';          % attributes
                                           'F2'});        % attributes
opts.network.metrics = [1];

%% Training options
opts.train = struct;
opts.train.batchSize = 20;
opts.train.numSubBatches = 1;
opts.train.numEpochs = 10000;
opts.train.continue = true;
opts.train.gpus = [1 2];
opts.train.learningRate = 0.001;
opts.train.weightDecay = 0.0005;
opts.train.expDir = opts.expDir;
opts.train.solver = @solver.adadelta;
opts.train.saveSolverState = true;
opts.train.randomSeed = 0;
opts.train.plotStatistics = true;
opts.train.nesterovUpdate = true ;
% DO NOT EDIT
opts.train.prefetch = opts.loadParallel;
opts.train.derOutputs = {opts.network.lossTypes(opts.network.loss),1};

%% Jitter options
opts.jitter = struct;
opts.jitter.randomCrop = true; % TODO: Randomly crop during training
opts.jitter.flipHorizontal = true; % Randomly flip horizontally during training
opts.jitter.flipVertical = true; % Randomly flip vertically during training

%% Image loading and preprocessing options
opts.prep = struct;
opts.prep.channels = [1 2 3]; % Specify color channels to be used
opts.prep.imgSize = [512 512]; % Dimensions must be a power of 2
opts.prep.lblSize = [384 384];
opts.prep.holdout = 0.1; % Proportion of images used for validation during training
opts.prep.preCrop = false; % Should images be cropped prior to training
opts.prep.preLoad = true; % true = images stored in memory
                           % false = images loaded from hard drive
% thresholding - set to empty for no thresholding. Otherwise, select a
%                value for binary thresholding
%              - for classification, set to bin edges just like the
%                histcounts function. There should be C+1 values, so if you
%                have 3 classes there should be 4 values.
opts.prep.binThreshold = [0.5 1.5 2.5 3.5];
opts.prep.foreVal = 1; % for attributes
opts.prep.backVal = -1; % for attributes
opts.prep.foreWeightType = 'distance'; % Can be distance or intensity
opts.prep.backWeightType = 'distance'; % Can be distance or intensity
opts.prep.foreWeightThresh = 250; % Threshold the foreground weights
opts.prep.backWeightThresh = 10;
opts.prep.foreWeightAmp = 1/3;
opts.prep.backWeightAmp = -0.5;
opts.prep.foreWeightThreshDir = -1; % 1 for positive threshold, -1 for negative threshold
opts.prep.backWeightThreshDir = 1;
opts.prep.windowSize = 1001; % Value must be odd
opts.prep.normImage = true;
opts.prep.normWeights = false;
opts.prep.maxNorm = 6;
opts.prep.maxWeight = 6;
opts.prep.mixWeightsWithLabels = false;
opts.prep.getBatch = @(imdb,batch) get_batch(imdb,batch,opts);
opts.prep.formatFunc = @() format_data(opts);

end