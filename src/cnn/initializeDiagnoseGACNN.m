function nn = initializeDiagnoseGACNN(nn,opts)
%% Initialize network and parameters
% nn is a trained discriminitive networks

% load an uninitialized network to generate a parallel gan
net = initializeDiagnoseCNN(opts);

%% Create attribute to detect real vs fake images
currentLayer = 'classifier_cat';
nn.addLayer(currentLayer,dagnn.Concat(),{'mutation_do','lobe_do','cadasil_do'},currentLayer);
previousLayer = currentLayer;

currentLayer = 'is_generated';
conv = dagnn.Conv('size',[8 8 1536 1],...
                  'pad',0,'stride',1,'hasBias',true);
nn.addLayer(currentLayer,conv,...
             {previousLayer},{currentLayer},...
             {[currentLayer '_f'] [currentLayer '_b']});
nn = init_params(nn);
         
l = dagnn.Loss('loss','logistic');
nn.addLayer('RealObj', l, {'is_generated','is_generated_label'}, 'RealObj');

%% Generate a parallel discriminator network for the the GAN
skipLayers = {'mutation_conv',...
              'mutation_pred',...
              'MutationObj',...
              'MutationErr',...
              'lobe_conv',...
              'lobe_pred',...
              'LobeObj',...
              'LobeErr',...
              'cadasil_pred',...
              'CadasilObj',...
              'CadasilErr'};

for i = 1:numel(net.layers)
    l = net.layers(i);
    if any(strcmp(l.name,skipLayers))
        continue;
    end
    l.name = ['gan_' l.name];
    for j = 1:numel(inputs)
        l.inputs{j} = ['gan_' l.inputs{j}];
    end
    for j = 1:numel(outputs)
        l.outputs{j} = ['gan_' l.outputs{j}];
    end
    if strcmp(class(l),{'dagnn.Conv' 'dagnn.ReLU' 'dagnn.BatchNorm'})
        nn.addLayer(l.name,l.block,...
                    l.inputs,l.outputs,...
                    l.params);
    else
        nn.addLayer(l.name,l.block,...
                    l.inputs,l.outputs);
    end
end

currentLayer = 'gan_classifier_cat';
nn.addLayer(currentLayer,dagnn.Concat(),{'gan_mutation_do','gan_lobe_do','gan_cadasil_do'},currentLayer);
previousLayer = currentLayer;

currentLayer = 'is_generated';
conv = dagnn.Conv('size',[8 8 1536 1],...
                  'pad',0,'stride',1,'hasBias',true);
nn.addLayer(['gan_' currentLayer],conv,...
             {previousLayer},{['gan_' currentLayer]},...
             {[currentLayer '_f'] [currentLayer '_b']});
nn = init_params(nn);
         
l = dagnn.Loss('loss','logistic');
nn.addLayer('FakeObj', l, {'gan_is_generated','gan_is_generated_label'}, 'FakeObj');

%% GAN Layer 1
% Input - 1x1x14
% Output = 2x2x64

% Concatenate class inputs
currentLayer = 'gan_generator';
nn.addLayer(currentLayer,dagnn.Concat(),{'noise','cadasil_dist','lobe_dist','mutation_dist'},currentLayer);
previousLayer = currentLayer;

% Convolution transpose
currentLayer = 'gan_ct1';
filters = single(bilinear_u(6, 14, 14)) ;
convT = dagnn.ConvTranspose('size', size(filters),'upsample', 2, 'crop', 2, ...
                            'numGroups', 14, 'hasBias', false);
nn.addLayer([currentLayer '_ct'],convT,...
            {currentLayer},{[currentLayer '_ct']},{[currentLayer '_ct_f']});
nn = init_params(nn);

f = nn.getParamIndex([currentLayer '_ct_f']) ;
nn.params(f).value = filters ;
nn.params(f).learningRate = 0 ;
nn.params(f).weightDecay = 1 ;
previousLayer = currentLayer;

% Convolution
currentLayer = 'gan_c1';
c = dagnn.Conv('size',[3 3 14 64],'pad',1,'stride',1,'hasBias',true);
nn.addLayer(currentLayer,c,{previousLayer},{currentLayer},...
           {[currentLayer '_f'] [currentLayer '_b']});
nn = init_params(nn);
nn.addLayer([currentLayer '_elu'],dagnn.ReLU('leak',opts.network.leakyReLU,'method',opts.network.reluMethod),...
            {currentLayer},{[currentLayer '_elu']});
currentLayer = [currentLayer '_elu'];
nn = init_params(nn);

%% GAN Layer 2
% Input - 2x2x64
% Output = 8x8x256

% Concatenate class inputs
currentLayer = 'gan_generator';
nn.addLayer(currentLayer,dagnn.Concat(),{'noise','cadasil_dist','lobe_dist','mutation_dist'},currentLayer);
previousLayer = currentLayer;

% Convolution transpose
currentLayer = 'gan_ct2';
filters = single(bilinear_u(12, 64, 64)) ;
convT = dagnn.ConvTranspose('size', size(filters),'upsample', 4, 'crop', 4, ...
                            'numGroups', 64, 'hasBias', false);
nn.addLayer([currentLayer '_ct'],convT,...
            {currentLayer},{[currentLayer '_ct']},{[currentLayer '_ct_f']});
nn = init_params(nn);

f = nn.getParamIndex([currentLayer '_ct_f']) ;
nn.params(f).value = filters ;
nn.params(f).learningRate = 0 ;
nn.params(f).weightDecay = 1 ;
previousLayer = currentLayer;

% Convolution
currentLayer = 'gan_c2';
c = dagnn.Conv('size',[3 3 64 256],'pad',1,'stride',1,'hasBias',true);
nn.addLayer(currentLayer,c,{previousLayer},{currentLayer},...
           {[currentLayer '_f'] [currentLayer '_b']});
nn = init_params(nn);
nn.addLayer([currentLayer '_elu'],dagnn.ReLU('leak',opts.network.leakyReLU,'method',opts.network.reluMethod),...
            {currentLayer},{[currentLayer '_elu']});
currentLayer = [currentLayer '_elu'];
nn = init_params(nn);

%%

nn = dagnn.DagNN();
previousLayer = 'input';
previousLayerSize = 3;
downLayers = {};

layers = 1:5;
layers = [layers 5-1:-1:1];

%% Downscaling layer 1
% Input 512x512x3
% Output 256x256x16
currentLayer = 'l1';
filterSize = [3,3,previousLayerSize,16];
[nn,currentLayer] = createRRLayer(nn,previousLayer,currentLayer,...
                                  filterSize(3),filterSize(4),...
                                  filterSize(1),8,...
                                  opts.network.reluMethod,opts.network.leakyReLU);
nn.layers(end-3).block.stride = [2 1];
nn.layers(end-3).block.pad = [1 1 0 0];
nn.layers(end-1).block.stride = [1 2];
nn.layers(end-1).block.pad = [0 0 1 1];
previousLayer = currentLayer;
previousLayerSize = 16;

%% Downscaling Layer 2
% Input 256x256x16
% Output 128x128x32
currentLayer = 'l2';
filterSize = [3,3,previousLayerSize,16];
[nn,currentLayer] = createRRLayer(nn,previousLayer,currentLayer,...
                                  filterSize(3),filterSize(4),...
                                  filterSize(1),8,...
                                  opts.network.reluMethod,opts.network.leakyReLU);
nn.layers(end-3).block.stride = [2 1];
nn.layers(end-3).block.pad = [1 1 0 0];
nn.layers(end-1).block.stride = [1 2];
nn.layers(end-1).block.pad = [0 0 1 1];
previousLayer = currentLayer;
previousLayerSize = 16;

%% Downscaling Layer 3
% Input 128x128x32
% Output 64x64x64
currentLayer = 'l3';
filterSize = [3,3,previousLayerSize,32];
[nn,currentLayer] = createRRLayer(nn,previousLayer,currentLayer,...
                                  filterSize(3),filterSize(4),...
                                  filterSize(1),8,...
                                  opts.network.reluMethod,opts.network.leakyReLU);
nn.layers(end-3).block.stride = [2 1];
nn.layers(end-3).block.pad = [1 1 0 0];
nn.layers(end-1).block.stride = [1 2];
nn.layers(end-1).block.pad = [0 0 1 1];
previousLayer = currentLayer;

currentLayer = 'l3_bn';
bn = dagnn.BatchNorm('numChannels',32);
nn.addLayer([currentLayer],bn,{previousLayer},{currentLayer},...
           {[currentLayer '_f'] [currentLayer '_b'] [currentLayer '_m']});
nn = init_params(nn);
previousLayer = currentLayer;
previousLayerSize = 32;

%% Downscaling Layer 4
% Input 64x64x64
% Output 32x32x128
currentLayer = 'l4';
filterSize = [3,3,previousLayerSize,64];
[nn,currentLayer] = createRRLayer(nn,previousLayer,currentLayer,...
                                  filterSize(3),filterSize(4),...
                                  filterSize(1),16,...
                                  opts.network.reluMethod,opts.network.leakyReLU);
nn.layers(end-3).block.stride = [2 1];
nn.layers(end-3).block.pad = [1 1 0 0];
nn.layers(end-1).block.stride = [1 2];
nn.layers(end-1).block.pad = [0 0 1 1];
previousLayer = currentLayer;

currentLayer = 'l4_bn';
bn = dagnn.BatchNorm('numChannels',64);
nn.addLayer([currentLayer],bn,{previousLayer},{currentLayer},...
           {[currentLayer '_f'] [currentLayer '_b'] [currentLayer '_m']});
nn = init_params(nn);
previousLayer = currentLayer;
previousLayerSize = 64;

%% Layer 5, Inception 1
% Input 32x32x128
% Output 32x32x128
iLayer = {};

currentLayer = 'l5_i1_c1';
c = dagnn.Conv('size',[1 1 previousLayerSize 32],'pad',0,'stride',1,'hasBias',true);
nn.addLayer(currentLayer,c,{previousLayer},{currentLayer},...
           {[currentLayer '_f'] [currentLayer '_b']});
nn = init_params(nn);
nn.addLayer([currentLayer '_elu'],dagnn.ReLU('leak',opts.network.leakyReLU,'method',opts.network.reluMethod),...
            {currentLayer},{[currentLayer '_elu']});
nn = init_params(nn);
iLayer{end+1} = [currentLayer '_elu'];

currentLayer = 'l5_i1_c3';
[nn,currentLayer] = createRRLayer(nn,previousLayer,currentLayer,...
                                  previousLayerSize,32,...
                                  3,8,...
                                  opts.network.reluMethod,opts.network.leakyReLU);
iLayer{end+1} = currentLayer;

currentLayer = 'l5_i1_c5';
[nn,currentLayer] = createRRLayer(nn,previousLayer,currentLayer,...
                                  previousLayerSize,32,...
                                  5,8,...
                                  opts.network.reluMethod,opts.network.leakyReLU);
iLayer{end+1} = currentLayer;

currentLayer = 'l5_i1_c';
c = dagnn.Conv('size',[1 1 previousLayerSize 32],'pad',0,'stride',1,'hasBias',true);
nn.addLayer(currentLayer,c,{previousLayer},{currentLayer},...
           {[currentLayer '_f'] [currentLayer '_b']});
nn = init_params(nn);
nn.addLayer([currentLayer '_elu'],dagnn.ReLU('leak',opts.network.leakyReLU,'method',opts.network.reluMethod),...
            {currentLayer},{[currentLayer '_elu']});
nn = init_params(nn);
maxPool = dagnn.Pooling('method', 'max', 'poolSize', [3 3],'pad', 1, 'stride', 1);
nn.addLayer([currentLayer '_m'],maxPool,...
            {[currentLayer '_elu']},{[currentLayer '_m']});
iLayer{end+1} = [currentLayer '_m'];

currentLayer = 'l5_i1_cat';
nn.addLayer(currentLayer,dagnn.Concat(),iLayer,currentLayer);
previousLayer = currentLayer;

currentLayer = 'l5_i1_bn';
bn = dagnn.BatchNorm('numChannels',128);
nn.addLayer([currentLayer],bn,{previousLayer},{currentLayer},...
           {[currentLayer '_f'] [currentLayer '_b'] [currentLayer '_m']});
nn = init_params(nn);
previousLayer = currentLayer;

%% Layer 5, Inception 2
% Input 32x32x128
% Output 32x32x256
iLayer = {};

currentLayer = 'l5_i2_c1';
c = dagnn.Conv('size',[1 1 previousLayerSize 64],'pad',0,'stride',1,'hasBias',true);
nn.addLayer(currentLayer,c,{previousLayer},{currentLayer},...
           {[currentLayer '_f'] [currentLayer '_b']});
nn = init_params(nn);
nn.addLayer([currentLayer '_elu'],dagnn.ReLU('leak',opts.network.leakyReLU,'method',opts.network.reluMethod),...
            {currentLayer},{[currentLayer '_elu']});
nn = init_params(nn);
iLayer{end+1} = [currentLayer '_elu'];

currentLayer = 'l5_i2_c3';
[nn,currentLayer] = createRRLayer(nn,previousLayer,currentLayer,...
                                  previousLayerSize,64,...
                                  3,16,...
                                  opts.network.reluMethod,opts.network.leakyReLU);
iLayer{end+1} = currentLayer;

currentLayer = 'l5_i2_c5';
[nn,currentLayer] = createRRLayer(nn,previousLayer,currentLayer,...
                                  previousLayerSize,64,...
                                  5,16,...
                                  opts.network.reluMethod,opts.network.leakyReLU);
iLayer{end+1} = currentLayer;

currentLayer = 'l5_i2_c';
c = dagnn.Conv('size',[1 1 previousLayerSize 64],'pad',0,'stride',1,'hasBias',true);
nn.addLayer(currentLayer,c,{previousLayer},{currentLayer},...
           {[currentLayer '_f'] [currentLayer '_b']});
nn = init_params(nn);
nn.addLayer([currentLayer '_elu'],dagnn.ReLU('leak',opts.network.leakyReLU,'method',opts.network.reluMethod),...
            {currentLayer},{[currentLayer '_elu']});
nn = init_params(nn);
maxPool = dagnn.Pooling('method', 'max', 'poolSize', [3 3],'pad', 1, 'stride', 1);
nn.addLayer([currentLayer '_m'],maxPool,...
            {[currentLayer '_elu']},{[currentLayer '_m']});
iLayer{end+1} = [currentLayer '_m'];

currentLayer = 'l5_i2_cat';
nn.addLayer(currentLayer,dagnn.Concat(),iLayer,currentLayer);
previousLayer = currentLayer;

currentLayer = 'l5_i2_m';
maxPool = dagnn.Pooling('method', 'max', 'poolSize', [2 2],'pad', 0, 'stride', 2);
nn.addLayer(currentLayer,maxPool,...
            {previousLayer},{currentLayer});
previousLayer = currentLayer;
previousLayerSize = 256;

currentLayer = 'l5_i2_bn';
bn = dagnn.BatchNorm('numChannels',256);
nn.addLayer([currentLayer],bn,{previousLayer},{currentLayer},...
           {[currentLayer '_f'] [currentLayer '_b'] [currentLayer '_m']});
nn = init_params(nn);
previousLayer = currentLayer;

%% Layer 6, Inception 1
% Input 16x16x256
% Output 16x16x256
iLayer = {};

currentLayer = 'l6_i1_c1';
c = dagnn.Conv('size',[1 1 previousLayerSize 64],'pad',0,'stride',1,'hasBias',true);
nn.addLayer(currentLayer,c,{previousLayer},{currentLayer},...
           {[currentLayer '_f'] [currentLayer '_b']});
nn = init_params(nn);
nn.addLayer([currentLayer '_elu'],dagnn.ReLU('leak',opts.network.leakyReLU,'method',opts.network.reluMethod),...
            {currentLayer},{[currentLayer '_elu']});
nn = init_params(nn);
iLayer{end+1} = [currentLayer '_elu'];

currentLayer = 'l6_i1_c3';
[nn,currentLayer] = createRRLayer(nn,previousLayer,currentLayer,...
                                  previousLayerSize,64,...
                                  3,16,...
                                  opts.network.reluMethod,opts.network.leakyReLU);
iLayer{end+1} = currentLayer;

currentLayer = 'l6_i1_c5';
[nn,currentLayer] = createRRLayer(nn,previousLayer,currentLayer,...
                                  previousLayerSize,64,...
                                  5,16,...
                                  opts.network.reluMethod,opts.network.leakyReLU);
iLayer{end+1} = currentLayer;

currentLayer = 'l6_i1_c';
c = dagnn.Conv('size',[1 1 previousLayerSize 64],'pad',0,'stride',1,'hasBias',true);
nn.addLayer(currentLayer,c,{previousLayer},{currentLayer},...
           {[currentLayer '_f'] [currentLayer '_b']});
nn = init_params(nn);
nn.addLayer([currentLayer '_elu'],dagnn.ReLU('leak',opts.network.leakyReLU,'method',opts.network.reluMethod),...
            {currentLayer},{[currentLayer '_elu']});
nn = init_params(nn);
maxPool = dagnn.Pooling('method', 'max', 'poolSize', [3 3],'pad', 1, 'stride', 1);
nn.addLayer([currentLayer '_m'],maxPool,...
            {[currentLayer '_elu']},{[currentLayer '_m']});
iLayer{end+1} = [currentLayer '_m'];

currentLayer = 'l6_i1_cat';
nn.addLayer(currentLayer,dagnn.Concat(),iLayer,currentLayer);
previousLayer = currentLayer;

currentLayer = 'l6_i1_bn';
bn = dagnn.BatchNorm('numChannels',256);
nn.addLayer([currentLayer],bn,{previousLayer},{currentLayer},...
           {[currentLayer '_f'] [currentLayer '_b'] [currentLayer '_m']});
nn = init_params(nn);
previousLayer = currentLayer;

%% Layer 6, Inception 2
% Input 16x16x256
% Output 8x8x512
iLayer = {};

currentLayer = 'l6_i2_c1';
c = dagnn.Conv('size',[1 1 previousLayerSize 128],'pad',0,'stride',1,'hasBias',true);
nn.addLayer(currentLayer,c,{previousLayer},{currentLayer},...
           {[currentLayer '_f'] [currentLayer '_b']});
nn = init_params(nn);
nn.addLayer([currentLayer '_elu'],dagnn.ReLU('leak',opts.network.leakyReLU,'method',opts.network.reluMethod),...
            {currentLayer},{[currentLayer '_elu']});
nn = init_params(nn);
iLayer{end+1} = [currentLayer '_elu'];

currentLayer = 'l6_i2_c3';
[nn,currentLayer] = createRRLayer(nn,previousLayer,currentLayer,...
                                  previousLayerSize,128,...
                                  3,32,...
                                  opts.network.reluMethod,opts.network.leakyReLU);
iLayer{end+1} = currentLayer;

currentLayer = 'l6_i2_c5';
[nn,currentLayer] = createRRLayer(nn,previousLayer,currentLayer,...
                                  previousLayerSize,128,...
                                  5,32,...
                                  opts.network.reluMethod,opts.network.leakyReLU);
iLayer{end+1} = currentLayer;

currentLayer = 'l6_i2_c';
c = dagnn.Conv('size',[1 1 previousLayerSize 128],'pad',0,'stride',1,'hasBias',true);
nn.addLayer(currentLayer,c,{previousLayer},{currentLayer},...
           {[currentLayer '_f'] [currentLayer '_b']});
nn = init_params(nn);
nn.addLayer([currentLayer '_elu'],dagnn.ReLU('leak',opts.network.leakyReLU,'method',opts.network.reluMethod),...
            {currentLayer},{[currentLayer '_elu']});
nn = init_params(nn);
maxPool = dagnn.Pooling('method', 'max', 'poolSize', [3 3],'pad', 1, 'stride', 1);
nn.addLayer([currentLayer '_m'],maxPool,...
            {[currentLayer '_elu']},{[currentLayer '_m']});
iLayer{end+1} = [currentLayer '_m'];

currentLayer = 'l6_i2_cat';
nn.addLayer(currentLayer,dagnn.Concat(),iLayer,currentLayer);
previousLayer = currentLayer;

currentLayer = 'l6_i2_m';
maxPool = dagnn.Pooling('method', 'max', 'poolSize', [2 2],'pad', 0, 'stride', 2);
nn.addLayer(currentLayer,maxPool,...
            {previousLayer},{currentLayer});
previousLayer = currentLayer;
previousLayerSize = 512;

currentLayer = 'l6_i2_bn';
bn = dagnn.BatchNorm('numChannels',512);
nn.addLayer([currentLayer],bn,{previousLayer},{currentLayer},...
           {[currentLayer '_f'] [currentLayer '_b'] [currentLayer '_m']});
nn = init_params(nn);
l6 = currentLayer;
previousLayer = l6;

%% CADASIL Prediction
previousLayer = l6;
nn.addLayer('cadasil_do1',dagnn.DropOut('rate',opts.network.useDropout),{previousLayer},{'cadasil_do1'});
previousLayer = 'cadasil_do1';
nn = init_params(nn);

currentLayer = 'cadasil_c1';
filterSize = [3,3,previousLayerSize,512];
[nn,currentLayer] = createRRLayer(nn,previousLayer,currentLayer,...
                                  filterSize(3),filterSize(4),...
                                  filterSize(1),128,...
                                  opts.network.reluMethod,opts.network.leakyReLU);
previousLayer = currentLayer;

% currentLayer = 'cadasil_c2';
% filterSize = [3,3,previousLayerSize,512];
% [nn,currentLayer] = createRRLayer(nn,previousLayer,currentLayer,...
%                                   filterSize(3),filterSize(4),...
%                                   filterSize(1),128,...
%                                   opts.network.reluMethod,opts.network.leakyReLU);
% previousLayer = currentLayer;

currentLayer = 'cadasil_bn';
bn = dagnn.BatchNorm('numChannels',512);
nn.addLayer([currentLayer],bn,{previousLayer},{currentLayer},...
           {[currentLayer '_f'] [currentLayer '_b'] [currentLayer '_m']});
nn = init_params(nn);
previousLayer = currentLayer;

nn.addLayer('cadasil_do',dagnn.DropOut('rate',opts.network.useDropout),{previousLayer},{'cadasil_do'});
previousLayer = 'cadasil_do';
nn = init_params(nn);

currentLayer = 'cadasil_pred';
conv = dagnn.Conv('size',[8 8 previousLayerSize 1],...
                  'pad',0,'stride',1,'hasBias',true);
nn.addLayer(currentLayer,conv,...
             {previousLayer},{currentLayer},...
             {[currentLayer '_f'] [currentLayer '_b']});
nn = init_params(nn);
         
l = dagnn.Loss('loss','logistic');
nn.addLayer('CadasilObj', l, {'cadasil_pred','cadasil_label','cadasil_weight'}, 'CadasilObj');
e = dagnn.Loss('loss','binaryerror');
nn.addLayer('CadasilErr',e,{'cadasil_pred','cadasil_label'},{'CadasilErr'});

%% Lobe Prediction
previousLayer = l6;
nn.addLayer('lobe_do1',dagnn.DropOut('rate',opts.network.useDropout),{previousLayer},{'lobe_do1'});
previousLayer = 'lobe_do1';
nn = init_params(nn);

currentLayer = 'lobe_c1';
filterSize = [3,3,previousLayerSize,512];
[nn,currentLayer] = createRRLayer(nn,previousLayer,currentLayer,...
                                  filterSize(3),filterSize(4),...
                                  filterSize(1),128,...
                                  opts.network.reluMethod,opts.network.leakyReLU);
previousLayer = currentLayer;

% currentLayer = 'lobe_c2';
% filterSize = [3,3,previousLayerSize,512];
% [nn,currentLayer] = createRRLayer(nn,previousLayer,currentLayer,...
%                                   filterSize(3),filterSize(4),...
%                                   filterSize(1),128,...
%                                   opts.network.reluMethod,opts.network.leakyReLU);
% previousLayer = currentLayer;

currentLayer = 'lobe_bn';
bn = dagnn.BatchNorm('numChannels',512);
nn.addLayer([currentLayer],bn,{previousLayer},{currentLayer},...
           {[currentLayer '_f'] [currentLayer '_b'] [currentLayer '_m']});
nn = init_params(nn);
previousLayer = currentLayer;

nn.addLayer('lobe_do',dagnn.DropOut('rate',opts.network.useDropout),{previousLayer},{'lobe_do'});
previousLayer = 'lobe_do';
nn = init_params(nn);

currentLayer = 'lobe_conv';
conv = dagnn.Conv('size',[8 8 previousLayerSize 3],...
                  'pad',0,'stride',1,'hasBias',true);
nn.addLayer(currentLayer,conv,...
            {previousLayer},{currentLayer},...
            {[currentLayer '_f'] [currentLayer '_b']});
nn = init_params(nn);
l = dagnn.SoftMax();
nn.addLayer('lobe_pred', l, 'lobe_conv', 'lobe_pred');
         
l = dagnn.Loss('loss','log');
nn.addLayer('LobeObj', l, {'lobe_pred','lobe_label','lobe_weight'}, 'LobeObj');
e = dagnn.Loss('loss','classerror');
nn.addLayer('LobeErr',e,{'lobe_pred','lobe_label'},{'LobeErr'});

%% Mutation Prediction
previousLayer = l6;
nn.addLayer('mutation_do1',dagnn.DropOut('rate',opts.network.useDropout),{previousLayer},{'mutation_do1'});
previousLayer = 'mutation_do1';
nn = init_params(nn);

currentLayer = 'mutation_c1';
filterSize = [3,3,previousLayerSize,512];
[nn,currentLayer] = createRRLayer(nn,previousLayer,currentLayer,...
                                  filterSize(3),filterSize(4),...
                                  filterSize(1),128,...
                                  opts.network.reluMethod,opts.network.leakyReLU);
previousLayer = currentLayer;

% currentLayer = 'mutation_c2';
% filterSize = [3,3,previousLayerSize,512];
% [nn,currentLayer] = createRRLayer(nn,previousLayer,currentLayer,...
%                                   filterSize(3),filterSize(4),...
%                                   filterSize(1),128,...
%                                   opts.network.reluMethod,opts.network.leakyReLU);
% previousLayer = currentLayer;

currentLayer = 'mutation_bn';
bn = dagnn.BatchNorm('numChannels',512);
nn.addLayer([currentLayer],bn,{previousLayer},{currentLayer},...
           {[currentLayer '_f'] [currentLayer '_b'] [currentLayer '_m']});
nn = init_params(nn);
previousLayer = currentLayer;

nn.addLayer('mutation_do',dagnn.DropOut('rate',opts.network.useDropout),{previousLayer},{'mutation_do'});
previousLayer = 'mutation_do';
nn = init_params(nn);

currentLayer = 'mutation_conv';
conv = dagnn.Conv('size',[8 8 previousLayerSize 3],...
                  'pad',0,'stride',1,'hasBias',true);
nn.addLayer(currentLayer,conv,...
            {previousLayer},{currentLayer},...
            {[currentLayer '_f'] [currentLayer '_b']});
nn = init_params(nn);
l = dagnn.SoftMax();
nn.addLayer('mutation_pred', l, 'mutation_conv', 'mutation_pred');
         
l = dagnn.Loss('loss','log');
nn.addLayer('MutationObj', l, {'mutation_pred','mutation_label','mutation_weight'}, 'MutationObj');
e = dagnn.Loss('loss','classerror');
nn.addLayer('MutationErr',e,{'mutation_pred','mutation_label'},{'MutationErr'});