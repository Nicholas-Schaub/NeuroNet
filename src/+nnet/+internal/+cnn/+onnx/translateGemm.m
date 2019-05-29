function NNTLayer = translateGemm(node,initializerDimMap,initializerRawDataMap,LayerName,ImportWeights,IsRecurrentNetwork, OpsetVersion)
%https://github.com/onnx/onnx/blob/master/docs/Operators.md#Gemm

%   Copyright 2018-2019 The MathWorks, Inc.

persistent GemmNum
if isempty(GemmNum)
    GemmNum = 0;
end

% Get the attributes
attributeNames = arrayfun(@(a) a.name, node.attribute,'UniformOutput',false);
alpha   = single(1);
beta    = single(1);
transA  = false;
transB  = false;
for t = 1:numel(attributeNames)
    switch attributeNames{t}
        case 'alpha'
            alpha  = single(node.attribute(t).f);
        case 'beta'
            beta =  single(node.attribute(t).f);
        case 'transA'
            transA = logical(node.attribute(t).i);
        case 'transB'
            transB = logical(node.attribute(t).i);
    end
end
if transA || ~transB
    warning(message('nnet_cnn_onnx:onnx:GemmTrans', LayerName));
    NNTLayer = [];
    return;
end

% Create FullyConnectedLayer
B           = node.input{2};        % Get weight matrix to determin numUnits
weightDim   = initializerDimMap(B);	% [F X]
numUnits    = double(weightDim(1));
FC          = fullyConnectedLayer(numUnits, 'Name', LayerName);

% Import weights
if ImportWeights
    C       = node.input{3};       % Bias
    biasDim = initializerDimMap(C);                     % Could be 1 or 2D. Supported shapes are [F] and [1 F]
    if ~biasDimOK(biasDim)
        warning(message('nnet_cnn_onnx:onnx:GemmBiasDim',LayerName));
        NNTLayer = [];
        return;
    end
    FC.Bias	= beta*single(initializerRawDataMap(C))';   % A column vector.
    if ~isKey(initializerRawDataMap, B)
        error(message('nnet_cnn_onnx:onnx:WeightHasNoInitializer', LayerName, B));
    end
    weight      = alpha*single(initializerRawDataMap(B));
    weightDim   = initializerDimMap(B);
    % Import rowmajor [F X] to colmajor [F X]
    weight      = reshape(weight, fliplr(weightDim)); 	% Now it's colmajor [X F].
    FC.Weights  = weight';                              % Now it's colmajor [F X].
end

% Return layer(s). If it's not an RNN, we need to prepend a Flatten layer.
if IsRecurrentNetwork
    NNTLayer = FC;
else
    GemmNum = GemmNum + 1;
    FL = nnet.onnx.layer.FlattenLayer([LayerName '_Flatten' num2str(GemmNum)]);
    NNTLayer = [FL, FC];
end
end

function tf = biasDimOK(biasDim)
% Could be 1 or 2D. Supported shapes are [F] and [1 F]
tf = numel(biasDim)==1 || numel(biasDim)==2 && biasDim(1)==1;
end
