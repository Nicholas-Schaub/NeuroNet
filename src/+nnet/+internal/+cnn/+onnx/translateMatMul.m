function NNTLayer = translateMatMul(node,LayerName,initializerDimMap,initializerRawDataMap,ImportWeights, OpsetVersion)
%   Copyright 2018 The MathWorks, Inc.

% Y = A*B. Only supported when A is 2-D. (B can be 1D or 2D).
A = node.input{1};
B = node.input{2};

% B must be an initializer
if ~isKey(initializerDimMap, B)
    warning(message('nnet_cnn_onnx:onnx:OneNodeOneConst', 'MatMul', LayerName));
    NNTLayer = [];
    return;
end
sizeB = initializerDimMap(B);
% Extend sizeB to 2D if needed
if numel(sizeB) == 1
    sizeB(2) = 1;
end
numUnits = sizeB(2);
% Create FullyConnectedLayer
FC = fullyConnectedLayer(numUnits, 'Name', LayerName);
% Set weights
if ImportWeights
    if ~isKey(initializerRawDataMap, B)
        error(message('nnet_cnn_onnx:onnx:WeightHasNoInitializer', LayerName, B));
    end
    FC.Bias     = zeros(numUnits, 1);
    W           = single(initializerRawDataMap(B));	% W is sizeB rowmajor, e.g., [C K].
    FC.Weights  = reshape(W, fliplr(sizeB));        % Now it's [K C] colmajor as needed by FC.
end
NNTLayer = FC;
end