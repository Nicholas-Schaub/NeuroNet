function NNTLayer = translateFlatten(node, LayerName, OpsetVersion)
%https://github.com/onnx/onnx/blob/master/docs/Operators.md#Flatten

% Get 'axis' attribute
attributeNames  = arrayfun(@(a) a.name, node.attribute,'UniformOutput',false);
attributeInt    = arrayfun(@(a) a.i, node.attribute,'UniformOutput',false);
if ~isempty(attributeNames)
    attributeIntMap = containers.Map(attributeNames, attributeInt);
end
axis = 1;   % ONNX default
if ismember('axis', attributeNames)
    axis =  double(attributeIntMap('axis'));
end
if axis~=1
    warning(message('nnet_cnn_onnx:onnx:FlattenAxis', LayerName));
    NNTLayer = [];
    return;
end

% Create the layer
NNTLayer = nnet.onnx.layer.FlattenLayer(LayerName);
end
