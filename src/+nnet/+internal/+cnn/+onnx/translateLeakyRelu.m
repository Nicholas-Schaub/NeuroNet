function NNTLayer = translateLeakyRelu(thisNode,LayerName, OpsetVersion)
%Get the attributes
attributeNames = arrayfun(@(a) a.name, thisNode.attribute,'UniformOutput',false);
attributeFs = arrayfun(@(a) a.f, thisNode.attribute,'UniformOutput',false);
if ~isempty(attributeNames)
    attributeMap = containers.Map(attributeNames, attributeFs);
end
alpha = .01;    % ONNX default
if ismember('alpha', attributeNames)
    alpha =  double(attributeMap('alpha'));
end
if ~isscalar(alpha)
    warning(message('nnet_cnn_onnx:onnx:NonScalarAlpha',LayerName));
    NNTLayer = [];
    return;
end
NNTLayer = leakyReluLayer(alpha, 'Name', LayerName);
end
