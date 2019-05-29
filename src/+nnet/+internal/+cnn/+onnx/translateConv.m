function NNTLayer = translateConv(node,LayerName,initializerDimMap,initializerRawDataMap,ImportWeights, OpsetVersion)

%Get the attributes
attributeNames = arrayfun(@(a) a.name, node.attribute,'UniformOutput',false);
attributeInts = arrayfun(@(a) a.ints, node.attribute,'UniformOutput',false);
attributeIntsMap = containers.Map(attributeNames, attributeInts);
attributeString = arrayfun(@(a) a.s, node.attribute,'UniformOutput',false);
attributeStringMap = containers.Map(attributeNames, attributeString);

[~,idx] = ismember('group', attributeNames) ;
numWtGroups = 1;
if idx>0
    numWtGroups = double(node.attribute(idx).i);
end
if ismember( 'strides', attributeNames)
    Stride =  double(attributeIntsMap('strides')); %[h w]
else
    Stride = [1 1];
end

% Handle padding
Padding = [0 0 0 0];
if ismember('auto_pad',attributeNames)
    auto_pad = attributeStringMap('auto_pad');
    switch auto_pad
        case 'SAME_UPPER'
            Padding = 'same';
        case 'SAME_LOWER'
            Padding = 'same';
            warning(message('nnet_cnn_onnx:onnx:AutoPadSameLower',LayerName));
        case 'VALID'
            Padding = [0 0 0 0];
    end
elseif ismember('pads', attributeNames)
    Padding = double(attributeIntsMap('pads'));
    %ONNX: [H_b,W_b,H_end,W_end] ==> [t l b r]
    %MATLAB: [t b l r]
    Padding = Padding([1,3,2,4]);
end

if ismember('dilations',attributeNames)
    dilations = double(attributeIntsMap('dilations'));
else
    dilations = [1 1];
end

FilterSize = double(attributeIntsMap('kernel_shape')); %[h w] matches with MATLAB

%To get number of filters
weight_name = node.input{2};
weightDim = initializerDimMap(weight_name); %NumFilters x NumChannels/NumGroups x H x W
if numel(weightDim) > 4
    warning(message('nnet_cnn_onnx:onnx:ConvDim',LayerName));
    NNTLayer = [];
    return;
end
NumFilters = double(weightDim(1));

if ImportWeights
    %assume the 2nd input is W and the 3rd input is bias.
    %to double check
    if numel(node.input) > 2
        biasName = node.input{3};
        bias = single(initializerRawDataMap(biasName));
        bias = reshape(bias, [1,1,NumFilters]);
    else
        bias = zeros(1,1,NumFilters, 'single');
    end
    
    weights = permute(reshape(single(initializerRawDataMap(weight_name)), fliplr(weightDim)),[2,1,3,4]);
    %reshaping the weight to [W,H,C,F]). in Python, rawdata is saved in a row major.
    %permute(...,[2,1,3,4]) ==> HWCF
    % MATLAB: Height-Width-NumChannels-NumFilters: H x W x C x F
    %Here C is the number of channels per group.
else
    % Set default: empty
    bias = [];
    weights = [];
end
    
if numWtGroups == 1
    % Create ordinary convolution
    layer = convolution2dLayer(FilterSize, NumFilters,...
        'Stride', Stride,...
        'DilationFactor', dilations, ...
        'Padding', Padding,...
        'Weights', weights, ...
        'Bias', bias, ...
        'Name', LayerName);
else
    % Create grouped convolution
    NumFiltersPerGroup = NumFilters/numWtGroups;    
    if ImportWeights
        NumChannelsPerGroup = double(weightDim(2));
        weights = reshape(weights, [FilterSize,NumChannelsPerGroup,...
            NumFiltersPerGroup,numWtGroups]);
        bias = reshape(bias, [1,1,NumFiltersPerGroup,numWtGroups]);
    end
    layer = groupedConvolution2dLayer(FilterSize, NumFiltersPerGroup, numWtGroups,...
        'Stride', Stride,...
        'DilationFactor', dilations, ...
        'Padding', Padding,...
        'Weights', weights, ...
        'Bias', bias, ...
        'Name', LayerName);
end

NNTLayer = layer;
end
