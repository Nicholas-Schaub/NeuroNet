function NNTLayer = translateUnsupportedONNXLayers(node,LayerName,initializerDimMap,initializerRawDataMap,initializerNames,ImportWeights, OpsetVersion)

if isempty(node.input)
    WeightList = [];
    numInputs = 1;
else
    inputWithoutInitializer = ~ismember(node.input,initializerNames);
    WeightList = ~inputWithoutInitializer;
    numInputs = max(1, sum(inputWithoutInitializer));
end
numOutputs = max(1, numel(node.output));

NNTLayer =  nnet.onnx.layer.PlaceholderLayer(LayerName, node, numInputs, numOutputs);
NNTLayer.Weights = [];
if ImportWeights %import weights if needed
    WeightNameList = node.input(WeightList);
    for i = 1:numel(WeightNameList)
        WeightName = WeightNameList{i};
        if isempty(initializerDimMap) || isempty(initializerRawDataMap)
            error(message('nnet_cnn_onnx:onnx:EmptyInitializer'));
        end
        WeightDim = initializerDimMap(WeightName);
        WeightValue = single(initializerRawDataMap(WeightName));
        nDim = numel(WeightDim);
        if nDim > 1
            WeightValue = reshape(WeightValue, fliplr(WeightDim));
            WeightValue = permute(WeightValue, nDim:-1:1);
        end
        NNTLayer.Weights(i).name = WeightName;
        NNTLayer.Weights(i).value = WeightValue;
    end
end
end