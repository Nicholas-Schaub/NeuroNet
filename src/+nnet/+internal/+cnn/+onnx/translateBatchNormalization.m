function NNTLayer = translateBatchNormalization(thisNode,LayerName,initializerDimMap,initializerRawDataMap,ImportWeights, OpsetVersion)
EpsilonDefault  = 1e-5;
EpsilonMin      = 1e-5;
Epsilon         = EpsilonDefault;
attributeNames  = arrayfun(@(a) a.name, thisNode.attribute,'UniformOutput',false);
for t =1:numel(attributeNames)
    switch attributeNames{t}
        case 'epsilon'
            Epsilon = thisNode.attribute(t).f;
            if Epsilon < single(1e-5)
                warning(message('nnet_cnn_onnx:onnx:BadEpsilon', LayerName));
            end
        case  'spatial'
            spatial = thisNode.attribute(t).i;
            if spatial == 0
                warning(message('nnet_cnn_onnx:onnx:PerChannelMean',LayerName));
                NNTLayer = [];
                return;
            end
    end
end
Epsilon = max(double(Epsilon), EpsilonMin);
if ImportWeights
    ScaleName       = thisNode.input{2};
    BiasName        = thisNode.input{3};
    meanName        = thisNode.input{4};
    varName         = thisNode.input{5};
    NumChannels     = double(initializerDimMap(ScaleName));
    TrainedMean     = reshape(single(initializerRawDataMap(meanName)), [1,1,NumChannels]);
    TrainedVariance = reshape(single(initializerRawDataMap(varName)), [1,1,NumChannels]);
    Offset          = single(reshape(single(initializerRawDataMap(BiasName)), [1,1,NumChannels]));
    Scale           = reshape(single(initializerRawDataMap(ScaleName)), [1,1,NumChannels]);
    NNTLayer        = batchNormalizationLayer('Name', LayerName, 'Offset', Offset,...
        'Scale', Scale, 'Epsilon', Epsilon, 'TrainedMean', TrainedMean, ...
        'TrainedVariance', TrainedVariance);
else
    NNTLayer = batchNormalizationLayer('Name', LayerName, 'Epsilon', Epsilon);
end
end