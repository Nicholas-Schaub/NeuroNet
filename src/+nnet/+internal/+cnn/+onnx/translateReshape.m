function NNTLayer = translateReshape(node, initializerDimMap, initializerRawDataMap, LayerName, OpsetVersion)
%https://github.com/onnx/onnx/blob/master/docs/Operators.md#Reshape

% Get the shape
if OpsetVersion < 5
    % Get the shape from an attribute
    attributeNames      = arrayfun(@(a) a.name, node.attribute,'UniformOutput',false);
    attributeInts       = arrayfun(@(a) a.ints, node.attribute,'UniformOutput',false);
    attributeIntsMap    = containers.Map(attributeNames, attributeInts);
    shape = double(attributeIntsMap('shape'));
    shapeDim = numel(shape);
else
    % Opset >= 5
    % Shape is input 2
    if numel(node.input)~=2
        warning(message('nnet_cnn_onnx:onnx:ReshapeNumargs',LayerName));
        NNTLayer = [];
        return;
    end
    shapeName = node.input{2};
    if ~isKey(initializerDimMap, shapeName)
        warning(message('nnet_cnn_onnx:onnx:ReshapeArg2Constant',LayerName));
        NNTLayer = [];
        return;
    end
    shapeDim = initializerDimMap(shapeName);
    shape = double((initializerRawDataMap(shapeName)));
end
% Check target shape. To correspond to a flatten, shape must be length 2
% and shape(1) could be 0 or -1. We also see a case in inception_v1 in
% which the ONNX node reshapes to [1 1024], (which seems to incorrectly
% ignore the batch size). There may be other cases.
if ~(shapeDim==2 && ismember(shape(1), [-1 0 1]))
    warning(message('nnet_cnn_onnx:onnx:ReshapeFlatten', LayerName));
    NNTLayer = [];
    return;
end
% Create the layer
NNTLayer = nnet.onnx.layer.FlattenLayer(LayerName);
end
