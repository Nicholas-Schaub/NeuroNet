function NNTLayerGraphOrArray  = translateONNX(modelProto, OutputLayerType, UserImageInputSize, ImportWeights)

%   Copyright 2018 The MathWorks, Inc.

modelProto = iMakeNodeNamesDLTCompatible(modelProto);
thisGraph = modelProto.graph;
% Find opset version
if ~isempty(modelProto.opset_import)
    OpsetVersion = max(double([modelProto.opset_import.version]));
else
    OpsetVersion = 1;
end

% Get initializers
initializerNames    = arrayfun(@(a) a.name, thisGraph.initializer,'UniformOutput',false);
initializerDims     = arrayfun(@(a) a.dims, thisGraph.initializer,'UniformOutput',false);
initializerRawData  = getInitializerData(thisGraph);

% If any initializers have empty 'dims' property, set it to a 2D row vector
% representing the raw data size
emptyDims = cellfun(@isempty, initializerDims);
initializerDims(emptyDims) = cellfun(@(d)int64([1 numel(d)]), initializerRawData(emptyDims), 'UniformOutput', false);

if ~isempty(initializerNames)
    initializerDimMap       = containers.Map(initializerNames,initializerDims);
    initializerRawDataMap   = containers.Map(initializerNames,initializerRawData);
    clear initializerRawData    %save memeory
else
    initializerDimMap       = containers.Map;
    initializerRawDataMap   = containers.Map;
end

% Check number of inputs and outputs
if numel(thisGraph.output) > 1
    error(message('nnet_cnn_onnx:onnx:MO'));
end
inputsWithoutInitializers = {thisGraph.input.name};
if ~isempty(thisGraph.initializer)
    inputsWithoutInitializers = setdiff(inputsWithoutInitializers, {thisGraph.initializer.name});
end
if numel(inputsWithoutInitializers) > 1
    error(message('nnet_cnn_onnx:onnx:MI'));
end

% Replace weight-producing nodes with initializers (to the extent possible)
[thisGraph, initializerNames, initializerDimMap, initializerRawDataMap] = iReplaceWeightNodes(...
    thisGraph, initializerNames, initializerDimMap, initializerRawDataMap, OpsetVersion);

% Add the input layer first
NNTLayers = { createInputLayer(thisGraph, inputsWithoutInitializers{1}, UserImageInputSize) };
IsRecurrentNetwork = isa(NNTLayers{1}, 'nnet.cnn.layer.SequenceInputLayer');

% Add remaining layers
for i=1:numel(thisGraph.node)
    thisNode = thisGraph.node(i);
    op_type = thisNode.op_type;
    LayerName  = thisNode.name;
    if isempty(LayerName)
        LayerName = strcat('node_',num2str(i));
    end
    layer = [];
    switch op_type
        case 'Add'
            layer = nnet.internal.cnn.onnx.translateAdd(thisNode,LayerName,initializerDimMap,initializerRawDataMap,initializerNames, OpsetVersion);
        case 'AveragePool'
            layer = nnet.internal.cnn.onnx.translateAveragePool(thisNode,LayerName, OpsetVersion);
        case 'BatchNormalization'
            layer = nnet.internal.cnn.onnx.translateBatchNormalization(thisNode,LayerName,initializerDimMap,initializerRawDataMap,ImportWeights, OpsetVersion);
        case 'Clip'
            layer = nnet.internal.cnn.onnx.translateClip(thisNode, LayerName, OpsetVersion);
        case 'Concat'
            layer = depthConcatenationLayer(numel(thisNode.input), 'Name', LayerName);
        case 'Conv'
            layer = nnet.internal.cnn.onnx.translateConv(thisNode,LayerName,initializerDimMap,initializerRawDataMap,ImportWeights, OpsetVersion);
        case 'ConvTranspose'
            layer = nnet.internal.cnn.onnx.translateConvTranspose(thisNode,LayerName,initializerDimMap,initializerRawDataMap,ImportWeights, OpsetVersion);
        case 'Div'
            layer = nnet.internal.cnn.onnx.translateDiv(thisNode,LayerName,initializerDimMap,initializerRawDataMap,initializerNames, OpsetVersion);
        case 'Dropout'
            layer = nnet.internal.cnn.onnx.translateDropout(thisNode,LayerName, OpsetVersion);
        case 'Flatten'
            layer = nnet.internal.cnn.onnx.translateFlatten(thisNode, LayerName, OpsetVersion);
        case 'Gemm'
            layer = nnet.internal.cnn.onnx.translateGemm(thisNode,initializerDimMap,initializerRawDataMap,LayerName,ImportWeights,IsRecurrentNetwork, OpsetVersion);
        case 'GlobalAveragePool'
            layer = nnet.onnx.layer.GlobalAveragePooling2dLayer(LayerName);
        case 'Identity'
            layer = nnet.onnx.layer.IdentityLayer(LayerName);
        case 'ImageScaler'
            layer = nnet.internal.cnn.onnx.translateImageScaler(thisNode, LayerName, OpsetVersion);
        case 'LeakyRelu'
            layer = nnet.internal.cnn.onnx.translateLeakyRelu(thisNode,LayerName, OpsetVersion);
        case 'LRN'
            layer = nnet.internal.cnn.onnx.translateLRN(thisNode,LayerName, OpsetVersion);
        case 'LSTM'
            layer = nnet.internal.cnn.onnx.translateLSTM(thisNode,LayerName,initializerDimMap,initializerRawDataMap,ImportWeights, OpsetVersion);
        case 'MatMul'
            layer = nnet.internal.cnn.onnx.translateMatMul(thisNode,LayerName,initializerDimMap,initializerRawDataMap,ImportWeights, OpsetVersion);
        case 'MaxPool'
            layer = nnet.internal.cnn.onnx.translateMaxPool(thisNode,LayerName, OpsetVersion);
        case 'Mul'
            layer = nnet.internal.cnn.onnx.translateMul(thisNode,LayerName,initializerDimMap,initializerRawDataMap,initializerNames, OpsetVersion);
        case 'PRelu'
            layer = nnet.internal.cnn.onnx.translatePRelu(thisNode, initializerDimMap, initializerRawDataMap, LayerName, IsRecurrentNetwork, OpsetVersion);
        case 'Relu'
            layer = reluLayer('Name', LayerName);
        case 'Reshape'
            layer = nnet.internal.cnn.onnx.translateReshape(thisNode, initializerDimMap, initializerRawDataMap, LayerName, OpsetVersion);
        case 'Sigmoid'
            layer = nnet.onnx.layer.SigmoidLayer(LayerName);
        case 'Softmax'
            layer = softmaxLayer('Name', LayerName);
        case 'Tanh'
            layer = nnet.onnx.layer.TanhLayer(LayerName);
        case 'Sub'
            layer = nnet.internal.cnn.onnx.translateSub(thisNode,LayerName,initializerDimMap,initializerRawDataMap,initializerNames, OpsetVersion);
        case 'Sum'
            layer = additionLayer(numel(thisNode.input),'Name',LayerName);
    end
    if isempty(layer)
        layer = nnet.internal.cnn.onnx.translateUnsupportedONNXLayers(thisNode,LayerName,initializerDimMap,initializerRawDataMap,initializerNames,ImportWeights, OpsetVersion);
    end
    layer = maybeAppendOutputLayer(thisGraph, layer, OutputLayerType, thisNode);
    NNTLayers{end+1} = layer;
end
NNTLayers = NNTLayers';
if IsRecurrentNetwork
    NNTLayerGraphOrArray = [ NNTLayers{:} ]';
else
    NNTLayerGraphOrArray = makeConnections(NNTLayers, thisGraph, initializerNames);
end
end

function layer = maybeAppendOutputLayer(g,layer, OutputLayerType,thisNode)
%check if this layer's output is the output of the graph
%add outputLayer if this layer is the output
if ~isempty(OutputLayerType)
    [~,pos] = ismember(thisNode.output, g.output.name);
    if pos>0
        layer = [layer, createOutputLayer(layer, OutputLayerType, thisNode.op_type)];
    end
end
end

function NNTLayers = createOutputLayer(layer, OutputLayerType, op_type)
% Create an output layer, and optionally a preceding Softmax if the output
% mode is classification and the softmax layer is missing.
switch OutputLayerType
    case {'classification'}
        SM = maybeAddSoftmax(op_type, layer);
        NNTLayers = [SM, classificationLayer('Name', sprintf('ClassificationLayer_%s', layer(end).Name))];
    case {'regression'}
        NNTLayers = regressionLayer('Name', sprintf('RegressionLayer_%s', layer.Name));
    case {'pixelclassification'}
        SM = maybeAddSoftmax(op_type, layer);
        NNTLayers = [SM, pixelClassificationLayer('Name', sprintf('PixelClassificationLayer_%s', layer(end).Name))];
    otherwise
        assert(false);
end
end

function SM = maybeAddSoftmax(op_type, layer)
if isequal(op_type, 'Softmax')
    SM = [];
else
    warning(message('nnet_cnn_onnx:onnx:AddingSoftmax'));
    SM = softmaxLayer('Name', sprintf('SoftmaxLayer_%s', layer(end).Name));
end
end

function NNTLayerGraph = makeConnections(NNTLayers, thisGraph, initializerNames)
NNTLayerGraph = layerGraph();
destinationLayerNameMap = containers.Map;
for L = 1:numel(NNTLayers)
    if numel(NNTLayers{L}) > 1
        % The operator was translated to a series of NNT layers, the second
        % of which has the name of the ONNX operator.
        destinationLayerNameMap(NNTLayers{L}(2).Name) = NNTLayers{L}(1).Name;
    end
    NNTLayerGraph = addLayers(NNTLayerGraph, NNTLayers{L});
end
NNTLayerGraph = nnet.internal.cnn.onnx.addConnections(NNTLayerGraph, thisGraph, destinationLayerNameMap, initializerNames);
end

function inputLayer = createInputLayer(graph, inputTensorName, UserImageInputSize)
% Find the input tensor
[~,pos] = ismember(inputTensorName, {graph.input.name});
assert(pos>0);
% Get its shape, including possible 'None' entries
ONNXInputShape      = [graph.input(pos).type.tensor_type.shape.dim.dim_value];     % fchw for image input, or [seqLen, batchSize, inputDim] for sequence input.
ONNXInputSymbols	= {graph.input(pos).type.tensor_type.shape.dim.dim_param};     % Non-empty elements indicate undetermined sizes.
NNTInputName        = ['Input_', nnet.internal.cnn.onnx.makeNNTName(inputTensorName)];
switch numel(ONNXInputSymbols)
    case 3
        % It's a sequence input layer: [num_training_samples, max_length, num_features]
        NNTInputShape = double(ONNXInputShape(end));
        inputLayer = sequenceInputLayer(NNTInputShape, 'Name', NNTInputName);
    case 4
        % Image input layer
        if numel(ONNXInputShape)==3 && ~isempty(ONNXInputSymbols{1})
            % ONNXInputShape is really [None c h w]. Make it [1 c h w]
            ONNXInputShape = [1 ONNXInputShape(:)'];
        end
        NNTInputShape = double(ONNXInputShape([3 4 2]));    % Make it [h w c].
        inputLayer = imageInputLayer(NNTInputShape, 'Name', NNTInputName, 'Normalization', 'none');
    otherwise
        error(message('nnet_cnn_onnx:onnx:BadInputShape'));
end
end

function data = getInitializerData(graph)
% Return a cell array of the data for each initializer.
data = {};
for i = numel(graph.initializer):-1:1
    data{i} = getDataFromTensorProto(graph.initializer(i));
    if isempty(data{i})
        error(message('nnet_cnn_onnx:onnx:InitializerDataTypeUnsupported', graph.initializer(i).name,...
            char(graph.initializer(i).data_type)));
    end
end
end

function data = getDataFromTensorProto(tproto)
switch tproto.data_type
    % Types with dedicated fields
    case nnet.internal.cnn.onnx.TensorProto_DataType.FLOAT
        data = typecast(getDataFromTypedFieldOrRawData(tproto, 'float_data'), 'single');
    case nnet.internal.cnn.onnx.TensorProto_DataType.INT32
        data = typecast(getDataFromTypedFieldOrRawData(tproto, 'int32_data'), 'int32');
    case nnet.internal.cnn.onnx.TensorProto_DataType.STRING
        data = typecast(getDataFromTypedFieldOrRawData(tproto, 'string_data'), 'uint8');
    case nnet.internal.cnn.onnx.TensorProto_DataType.INT64
        data = typecast(getDataFromTypedFieldOrRawData(tproto, 'int64_data'), 'int64');
    case nnet.internal.cnn.onnx.TensorProto_DataType.DOUBLE
        data = typecast(getDataFromTypedFieldOrRawData(tproto, 'double_data'), 'double');
    case nnet.internal.cnn.onnx.TensorProto_DataType.UINT64
        data = typecast(getDataFromTypedFieldOrRawData(tproto, 'uint64_data'), 'uint64');
        % Other types
    case nnet.internal.cnn.onnx.TensorProto_DataType.INT8
        data = typecast(getDataFromTypedFieldOrRawData(tproto, 'raw_data'), 'int8');
    case nnet.internal.cnn.onnx.TensorProto_DataType.UINT16
        data = typecast(getDataFromTypedFieldOrRawData(tproto, 'raw_data'), 'uint16');
    case nnet.internal.cnn.onnx.TensorProto_DataType.INT16
        data = typecast(getDataFromTypedFieldOrRawData(tproto, 'raw_data'), 'int16');
    case nnet.internal.cnn.onnx.TensorProto_DataType.UINT32
        data = typecast(getDataFromTypedFieldOrRawData(tproto, 'raw_data'), 'uint32');
    otherwise
        data = [];
end
end

function data = getDataFromTypedFieldOrRawData(tproto, fieldName)
% Look in the field name and also in 'raw_data'
if ~isempty(tproto.(fieldName))
    data = tproto.(fieldName);
elseif ~isempty(tproto.raw_data)
    data = tproto.raw_data;
else
    error(message('nnet_cnn_onnx:onnx:InitializerDataNotFound', tproto.name));
end
end

function [graph, initializerNames, initializerDimMap, initializerRawDataMap] = iReplaceWeightNodes(...
    graph, initializerNames, initializerDimMap, initializerRawDataMap, OpsetVersion)
% For now just handle known feasible cases:
% (1) A Constant node
% (2) A Reshape node that takes an initializer as input.
% (3) An Unsqueeze node that takes an initializer as input.
toDelete = [];
for i = 1:numel(graph.node)
    node        = graph.node(i);
    op_type     = node.op_type;
    OutputName  = node.output{1};
    tf          = false;
    switch op_type
        case 'Constant'
            [tf, dim, rawData] = transformConstantNodeToInitializer(node);
        case 'Reshape'
            [tf, dim, rawData] = transformReshapeNodeToInitializer(node, initializerDimMap, initializerRawDataMap, OpsetVersion);
        case 'Unsqueeze'
            [tf, dim, rawData] = transformUnsqueezeNodeToInitializer(node, initializerDimMap, initializerRawDataMap, OpsetVersion);
    end
    if tf
        % Add the new initializer
        initializerNames{end+1}         = OutputName;
        initializerDimMap(OutputName)   = dim;
        initializerRawDataMap(OutputName)  = rawData;
        % Mark the node for deletion from the graph
        toDelete(end+1) = i;
    end
end
graph.node(toDelete) = [];
end

function modelProto = iMakeNodeNamesDLTCompatible(modelProto)
nodes = modelProto.graph.node;
for i=1:numel(nodes)
    if ~isempty(nodes(i).name)
        nodes(i).name = strrep(nodes(i).name, '/', '|');
    end
end
modelProto.graph.node = nodes;
end

%% Transforming certain nodes to initializers

function [tf, dim, rawData] = transformConstantNodeToInitializer(node)
%Get the tensor attributes
attributeNames = arrayfun(@(a) a.name, node.attribute,'UniformOutput',false);
attributeTensors = arrayfun(@(a) a.t, node.attribute,'UniformOutput',false);
attributeTensorsMap = containers.Map(attributeNames, attributeTensors);

assert(ismember('value', attributeNames));
tproto = attributeTensorsMap('value');
tf = true;
if isempty(tproto.dims)
    dim = 1;
else
    dim = double(tproto.dims);
end
rawData = double(getDataFromTensorProto(tproto));
end

function [tf, dim, rawData] = transformReshapeNodeToInitializer(node, initializerDimMap, initializerRawDataMap, OpsetVersion)
% If this node is reshaping an initializer, reshape it here and return it.
inputName = node.input{1};
if ~isKey(initializerDimMap, inputName)
    % The first input is not an initializer
    tf = false;
    dim = [];
    rawData = [];
else
    % The first input is an initializer.
    % Get the desired shape
    if OpsetVersion < 5
        % Get the shape from an attribute
        attributeNames      = arrayfun(@(a) a.name, node.attribute,'UniformOutput',false);
        attributeInts       = arrayfun(@(a) a.ints, node.attribute,'UniformOutput',false);
        attributeIntsMap    = containers.Map(attributeNames, attributeInts);
        shape               = double(attributeIntsMap('shape'));
        shapeDim          	= numel(shape);
    else
        % Opset >= 5
        % Shape is input 2
        if numel(node.input)~=2
            warning(message('nnet_cnn_onnx:onnx:ReshapeNumargs',node.name));
            NNTLayer = [];
            return;
        end
        shapeName   = node.input{2};
        shapeDim    = initializerDimMap(shapeName);
        shape       = double((initializerRawDataMap(shapeName)));
    end
    % Return the first input's raw data, and the desired shape
    tf = true;
    dim = shape;
    rawData = single(initializerRawDataMap(inputName));
end
end

function [tf, dim, rawData] = transformUnsqueezeNodeToInitializer(node, initializerDimMap, initializerRawDataMap, OpsetVersion)
% If this node is unsqueezing an initializer, unsqueeze it here and return it.
inputName = node.input{1};
if ~isKey(initializerDimMap, inputName)
    % The input is not an initializer
    tf = false;
    dim = [];
    rawData = [];
else
    % The input is an initializer.
    % Get axes
    attributeNames      = arrayfun(@(a) a.name, node.attribute,'UniformOutput',false);
    attributeInts       = arrayfun(@(a) a.ints, node.attribute,'UniformOutput',false);
    attributeIntsMap    = containers.Map(attributeNames, attributeInts);
    Axes                = double(attributeIntsMap('axes'));
    % Insert 1's into the shape
    shape         = initializerDimMap(inputName);
    shapeLen      = numel(Axes) + numel(shape);
    newShape      = ones(1,shapeLen);
    idx           = setdiff(1:shapeLen, Axes+1);
    newShape(idx) = shape;
    % Return the input raw data, and the desired shape
    tf      = true;
    dim     = newShape;
    rawData = single(initializerRawDataMap(inputName));
end
end