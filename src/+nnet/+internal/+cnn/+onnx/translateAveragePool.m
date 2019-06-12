function NNTLayer = translateAveragePool(node, LayerName, OpsetVersion)

%   Copyright 2018 The MathWorks, Inc.

%Get the attributes
attributeNames = arrayfun(@(a) a.name, node.attribute,'UniformOutput',false);
attributeInts = arrayfun(@(a) a.ints, node.attribute,'UniformOutput',false);
attributeIntsMap = containers.Map(attributeNames, attributeInts);
attributeInt        = arrayfun(@(a) a.i, node.attribute,'UniformOutput',false);
attributeIntMap     = containers.Map(attributeNames, attributeInt);
attributeString     = arrayfun(@(a) a.s, node.attribute,'UniformOutput',false);
attributeStringMap  = containers.Map(attributeNames, attributeString);

% strides
if ismember( 'strides', attributeNames)
    stride =  double(attributeIntsMap('strides')); %[h w]
else
    stride = [1 1];
end

% pads
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
    % ONNX:   [H_b,W_b,H_end,W_end] == [t l b r]
    %MATLAB: [t b l r]
    Padding = Padding([1 3 2 4]);
end

% count_include_pad
if OpsetVersion >= 7
    IncludePadInAvg = 0;
    if ismember('count_include_pad', attributeNames)
        IncludePadInAvg = double(attributeIntMap('count_include_pad'));
    end
    if IncludePadInAvg==0
        warning(message('nnet_cnn_onnx:onnx:AveragePoolingExcludesPadding', LayerName));
    end
end

% kernel_shape
poolSize = double(attributeIntsMap('kernel_shape'));

NNTLayer = averagePooling2dLayer(poolSize, 'Stride', stride, 'Padding', Padding, 'Name', LayerName);
end