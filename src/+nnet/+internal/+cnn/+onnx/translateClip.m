function NNTLayer = translateClip(thisNode,LayerName, OpsetVersion)

%Get the attributes
attributeNames = arrayfun(@(a) a.name, thisNode.attribute,'UniformOutput',false);
attributeFs = arrayfun(@(a) a.f, thisNode.attribute,'UniformOutput',false);
% Set Min and Max
Min = -Inf;
Max = Inf;
if ~isempty(attributeNames)
    attributeMap = containers.Map(attributeNames, attributeFs);
    if ismember('min', attributeNames)
        Min =  double(attributeMap('min'));
    end
    if ismember('max', attributeNames)
        Max =  double(attributeMap('max'));
    end
end
% Create the layer
NNTLayer = nnet.onnx.layer.ClipLayer(LayerName, Min, Max);
end