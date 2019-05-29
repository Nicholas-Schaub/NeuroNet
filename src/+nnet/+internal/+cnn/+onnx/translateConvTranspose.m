function NNTLayer = translateConvTranspose(node,LayerName,initializerDimMap,initializerRawDataMap,ImportWeights, OpsetVersion)

%Get the attributes
attributeNames = arrayfun(@(a) a.name, node.attribute,'UniformOutput',false);
attributeInts = arrayfun(@(a) a.ints, node.attribute,'UniformOutput',false);
attributeIntsMap = containers.Map(attributeNames, attributeInts);
attributeString = arrayfun(@(a) a.s, node.attribute,'UniformOutput',false);
attributeStringMap = containers.Map(attributeNames, attributeString);

% dilations
if ismember('dilations',attributeNames)
    dilations = double(attributeIntsMap('dilations'));
    if  ~all(dilations == 1)
        warning(message('nnet_cnn_onnx:onnx:UnsupportedDilation',LayerName));
        NNTLayer = [];
        return;
    end
end

% group
[~,idx] = ismember( 'group', attributeNames) ;
if idx>0
    numWtGroups = double(node.attribute(idx).i);
    if numWtGroups > 1
        warning(message('nnet_cnn_onnx:onnx:Max1WeightGroup',LayerName));
        NNTLayer = [];
        return;
    end
end

% pads
Padding = [0 0 0 0];
if ismember('auto_pad',attributeNames)
    auto_pad = attributeStringMap('auto_pad');
    switch auto_pad
        case 'SAME_UPPER'
            warning(message('nnet_cnn_onnx:onnx:ConvTransposeCropping',LayerName));
            NNTLayer = [];
            return;
        case 'SAME_LOWER'
            warning(message('nnet_cnn_onnx:onnx:ConvTransposeCropping',LayerName));
            NNTLayer = [];
            return;
        case 'VALID'
            Padding = [0 0 0 0];
    end
elseif ismember('pads', attributeNames)
    Padding = double(attributeIntsMap('pads'));
end
% Convert ONNX padding to DLT cropping:
% ONNX: [H_b,W_b,H_end,W_end] ==> [t l b r]
% MATLAB: [t=b l=r]
Cropping = Padding([1,3,2,4]);
if Cropping(1)~=Cropping(2) || Cropping(3)~=Cropping(4)
    warning(message('nnet_cnn_onnx:onnx:ConvTransposeCropping',LayerName));
    NNTLayer = [];
    return;
end
Cropping = Cropping([1 3]);

% strides
if ismember( 'strides', attributeNames)
    Stride = double(attributeIntsMap('strides')); %[h w]
else
    Stride = [1 1];
end

% output_padding and output_shape
if ismember('output_padding', attributeNames)
    warning(message('nnet_cnn_onnx:onnx:UnsupportedAttribute','output_padding',LayerName));
    NNTLayer = [];
    return;
end
if ismember('output_shape', attributeNames)
    warning(message('nnet_cnn_onnx:onnx:UnsupportedAttribute','output_shape',LayerName));
    NNTLayer = [];
    return;
end

% get FilterSize from weight tensor shape
weight_name = node.input{2};
weightDim   = initializerDimMap(weight_name); % CFHW
NumFilters 	= double(weightDim(2));
Height     	= double(weightDim(3));
Width     	= double(weightDim(4));
FilterSize 	= [Height, Width];

% Create the DLT layer
Conv2dTransposed = transposedConv2dLayer(FilterSize, NumFilters,...
    'Stride', Stride, 'Cropping', Cropping, 'Name', LayerName);

% Import weights
if ImportWeights
    % Get bias if it's an initializer
    if numel(node.input) > 2
        biasName = node.input{3};
        if isempty(initializerDimMap(biasName))
            warning(message('nnet_cnn_onnx:onnx:ConvTransposeBias', LayerName));
            NNTLayer = [];
            return;
        end
        bias = single(initializerRawDataMap(biasName));
        Conv2dTransposed.Bias = reshape(bias, [1,1,NumFilters]);
    else
        Conv2dTransposed.Bias = zeros(1,1,NumFilters, 'single');
    end
    % Get weights
    % ONNX:     CFHW row-major
    % MATLAB:   HWFC col-major
    W = single(initializerRawDataMap(weight_name));     % CFHW row-major
    W = reshape(W, fliplr(weightDim));                  % WHFC col-major
    W = permute(W, [2 1 3 4]);                          % HWFC col-major
    Conv2dTransposed.Weights = W;
end
NNTLayer = Conv2dTransposed;
end