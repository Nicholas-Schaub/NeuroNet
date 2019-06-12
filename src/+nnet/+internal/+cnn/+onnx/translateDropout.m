function NNTLayer = translateDropout(thisNode,LayerName, OpsetVersion)

%Get the attributes
attributeNames = arrayfun(@(a) a.name, thisNode.attribute,'UniformOutput',false);
attributeFs = arrayfun(@(a) a.f, thisNode.attribute,'UniformOutput',false);
if ~isempty(attributeNames)
    attributeMap = containers.Map(attributeNames, attributeFs);
end
rate = 0.5;     % ONNX default;
if ismember( 'ratio', attributeNames)
    rate =  double(attributeMap('ratio')); %ignore other fileds for now
end

NNTLayer = dropoutLayer(rate, 'Name', LayerName);
end
