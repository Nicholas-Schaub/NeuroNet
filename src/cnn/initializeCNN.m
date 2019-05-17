function nn = initializeCNN(opts)
%% Initialize network and parameters
nn = dagnn.DagNN();
previousLayer = 'input';
previousLayerSize = numel(opts.prep.channels);
downLayers = {};

layers = 1:opts.network.networkDepth;
if opts.network.type>3
    layers = [layers opts.network.networkDepth-1:-1:1];
end

%% Build layers
rng(opts.train.randomSeed);

for layer = 1:numel(layers)
    if layer==1
        isDownscale = true;
    elseif layers(layer)==max(layers)
        isDownscale = false;
    else
        isDownscale = (layers(layer) - layers(layer-1)) > 0;
    end
    
    complexity = 2^(opts.network.complexity+layers(layer)-1); 
    pad = floor(opts.network.filterSize/2);
    
    for repeat = 1:opts.network.layerDepth
        if isDownscale
            baseLayer = ['D' num2str(layers(layer)) '_R' num2str(repeat)];
        else
            baseLayer = ['U' num2str(layers(layer)) '_R' num2str(repeat)];
        end
        currentLayer = baseLayer;
    
        filterSize = [opts.network.filterSize,opts.network.filterSize,...
                      previousLayerSize,complexity];
        
        % Create layers
        switch opts.network.layerType
            case 1 % normal convolutional layer
                c = dagnn.Conv('size',filterSize,'pad',pad,'stride',1,'hasBias',true);
                nn.addLayer(currentLayer,c,{previousLayer},{currentLayer},...
                           {[currentLayer '_f'] [currentLayer '_b']});
                nn = init_params(nn);
                nn.addLayer([currentLayer '_relu'],dagnn.ReLU('leak',opts.network.leakyReLU,'method',opts.network.reluMethod),...
                            {currentLayer},{[currentLayer '_relu']});
                currentLayer = [currentLayer '_relu'];
                nn = init_params(nn);
                
            case 2 % reduced rank convolutional layer
                [nn,currentLayer] = createRRLayer(nn,previousLayer,currentLayer,...
                                     filterSize(3),filterSize(4),...
                                     filterSize(1),max(2*opts.network.complexity,filterSize(4)/8),...
                                     opts.network.reluMethod,opts.network.leakyReLU);
                                 
            case 3 % inception layer
                filtSizes = 1:2:filterSize(1);
                reductionLayers = (filterSize(4)/4)./filtSizes;
                reductionLayers(end+1) = reductionLayers(end);
                depth = reductionLayers.*2;
                [nn,currentLayer] = createInception(net,previousLayer,currentLayer,...
                                     filterSize(3),filtSizes,reductionLayers,depth,...
                                     opts.network.reluMethod,opts.network.leakyReLU);
        end

        % Create Network in Network layer (NIN)
        if opts.network.useNIN
            ninLayer = [baseLayer '_NIN'];
            c = dagnn.Conv('size',[1 1 filterSize(4) filterSize(4)],'pad',0,'stride',1,'hasBias',true);
            nn.addLayer(ninLayer,c,{currentLayer},{ninLayer},...
                       {[ninLayer '_f'] [ninLayer '_b']});
            nn = init_params(nn);
            nn.addLayer([ninLayer '_relu'],dagnn.ReLU('leak',opts.network.leakyReLU,'method',opts.network.reluMethod),...
                        {ninLayer},{[ninLayer '_relu']});
            nn = init_params(nn);
            currentLayer = [ninLayer '_relu'];
        end
        
        % Batch norm layer
        if opts.network.useBatchNorm && (layers(layer)/2 ~= round(layers(layer)/2))
            bn = dagnn.BatchNorm('numChannels',filterSize(4));
            nn.addLayer([baseLayer '_bn'],bn,{currentLayer},{[baseLayer '_bn']},...
                       {[baseLayer '_bn_f'] [baseLayer '_bn_b'] [baseLayer '_bn_m']});
            nn = init_params(nn);
            currentLayer = [baseLayer '_bn'];
        end
            

        % Residual layer
        if layer~=1
            if filterSize(3)~=filterSize(4)
                scaleLayer = [baseLayer '_scale'];
                c = dagnn.Conv('size',[1 1 filterSize(3) filterSize(4)],'pad',0,'stride',1,'hasBias',true);
                nn.addLayer(scaleLayer,c,{previousLayer},{scaleLayer},...
                           {[scaleLayer '_f'] [scaleLayer '_b']});
                nn = init_params(nn);
                nn.addLayer([scaleLayer '_relu'],dagnn.ReLU('leak',opts.network.leakyReLU,'method',opts.network.reluMethod),...
                            {scaleLayer},{[scaleLayer '_relu']});
                nn = init_params(nn);
                previousLayer = [scaleLayer '_relu'];
            end
            sumLayer = [baseLayer '_sum'];
            nn.addLayer(sumLayer,dagnn.Sum(),...
                        {currentLayer previousLayer},{sumLayer});
            nn = init_params(nn);
            currentLayer = sumLayer;
        end
        
        if repeat<opts.network.layerDepth
            previousLayer = currentLayer;
            previousLayerSize = filterSize(4);
        end
    end
    
    % Max Pooling or Convolutional Transpose layers
    if isDownscale
        downLayers{end+1} = currentLayer;
        maxPool = dagnn.Pooling('method', 'max', 'poolSize', [3 3],'pad', 1, 'stride', 2);
        nn.addLayer([baseLayer '_m'],maxPool,...
                    {currentLayer},{[baseLayer '_m']});
        previousLayer = [baseLayer '_m'];
        previousLayerSize = filterSize(4);
    elseif layer~=numel(layers)
        filters = single(bilinear_u(6, complexity, complexity)) ;
        convT = dagnn.ConvTranspose('size', size(filters),'upsample', 2, 'crop', 2, ...
                                    'numGroups', complexity/2, 'hasBias', false);
        nn.addLayer([baseLayer '_ct'],convT,...
                    {currentLayer},{[baseLayer '_ct']},{[baseLayer '_ct_f']});
        nn = init_params(nn);

        f = nn.getParamIndex([baseLayer '_ct_f']) ;
        nn.params(f).value = filters ;
        nn.params(f).learningRate = 0 ;
        nn.params(f).weightDecay = 1 ;

        nn.addLayer([baseLayer '_cat'],dagnn.Concat(),...
                   {[baseLayer '_ct'] downLayers{layers(layer+1)}},[baseLayer '_cat']);

        previousLayer = [baseLayer '_cat'];
        previousLayerSize = filterSize(4);
    else
        previousLayer = currentLayer;
        previousLayerSize = filterSize(4);
    end
end

%% Build outputs
% Averaging Layer
avgPool = dagnn.Pooling('method', 'avg', 'poolSize', filterSize(1),'pad', floor(filterSize(1)/2), 'stride', 1);
nn.addLayer('avg',avgPool,...
            {previousLayer},{'avg'});
previousLayer = 'avg';
        
% Dropout
if ~isempty(opts.network.useDropout)
    nn.addLayer('do',dagnn.DropOut('rate',opts.network.useDropout),{'avg'},{'do'});
    previousLayer = 'do';
    nn = init_params(nn);
end

% Prediction Layers
% Dense Layer
if opts.network.type>3
    convSize = opts.network.filterSize;
    pad = floor(convSize/2);
else
    convSize = opts.prep.imgSize(1)/2^(opts.network.networkDepth-1);
    pad = 0;
end
filterSize = [convSize convSize filterSize(4) opts.network.numClass];

if opts.network.type==2 || opts.network.type==3
    denseName = 'pred';
else
    denseName = 'dense';
end
dense = dagnn.Conv('size',filterSize,'pad',pad,'stride',1,'hasBias',true);
nn.addLayer('denseName',dense,{previousLayer},{denseName},{[denseName '_f'] [denseName '_b']});
nn = init_params(nn);
previousLayer = denseName;

% If pixel level analysis, crop the image to lblSize
if opts.network.type>3
    crop = dagnn.Crop();
    crop.crop = (opts.prep.imgSize-opts.prep.lblSize)./2;
    if opts.network.type~=4
        cropName = 'pred';
    else
        cropName = 'crop';
    end
    nn.addLayer(cropName,crop,{previousLayer 'label'},{cropName});
    nn = init_params(nn);
    previousLayer = cropName;
end

% If performing classification, use a softmax layer. Otherwise
if opts.network.type==1 || opts.network.type==4
    softmax = dagnn.SoftMax();
    nn.addLayer('pred',softmax,{previousLayer},{'pred'});
    nn = init_params(nn);
end

for i = 1:length(opts.network.metrics)
    switch opts.network.metricTypes(opts.network.metrics(i))
        case 'classerror'
            e = dagnn.Loss('loss','classerror');
            nn.addLayer('top1err',e,{'pred','label'},{'top1err'});
        case 'top5error'
            e = dagnn.Loss('loss','topkerror','topK',5);
            nn.addLayer('topkerr',e,{'pred','label'},{'top5err'});
        case 'binaryerror'
            e = dagnn.Loss('loss','binaryerror');
            nn.addLayer('binerr',e,{'pred','label'},{'binerr'});
        case 'F1'
            e = dagnn.Fscore();
            if mod(opts.network.type,3)==1
                e.type = 'class';
            end
            nn.addLayer('F1',e,{'pred','label'},{'F1'});
        case 'F2'
            e = dagnn.Fscore();
            e.beta = 2;
            if mod(opts.network.type,3)==0
                e.type = 'class';
            end
            nn.addLayer('F2',e,{'pred','label'},{'F2'});
        otherwise
            assert(false);
    end
end

switch opts.train.derOutputs{1}
    case 'L2'
        l = dagnn.L2Loss();
        nn.addLayer('L2', l, {'pred','label','weight'}, 'L2');
    case 'log'
        l = dagnn.Loss('loss','log');
        nn.addLayer('log', l, {'pred','label','weight'}, 'log');
    case 'logistic'
        l = dagnn.Loss('loss','logistic');
        nn.addLayer('logistic', l, {'pred','label','weight'}, 'logistic');
end