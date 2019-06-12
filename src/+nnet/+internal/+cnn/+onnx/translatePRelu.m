function NNTLayer = translatePRelu(node, initializerDimMap, initializerRawDataMap, LayerName, isRNN, OpsetVersion)

% Copyright 2018-2019 The Mathworks, Inc.

% slope is input 2
if numel(node.input)~=2
    warning(message('nnet_cnn_onnx:onnx:PReluNumargs',LayerName));
    NNTLayer = [];
    return;
end
% Import slope
slopeName	= node.input{2};
slope    	= double((initializerRawDataMap(slopeName)));
slopeDim 	= double((initializerDimMap(slopeName)));
if isRNN
    % ONNX X dim is [T N C]. Supported slopeDim can be [C] or [1 C] or [1 1 C]
    if ~RNNSlopeDimOK(slopeDim)
        warning(message('nnet_cnn_onnx:onnx:PReluSlopeDim',LayerName));
        NNTLayer = [];
        return;
    end
    numChannels   = numel(slope);
    DLTChannelDim = 1;              % [C N T]
else
    % ONNX X dim is [N C H W]. Supported slopeDim can be [C 1 1] or [1 C 1 1]
    if ~CNNSlopeDimOK(slopeDim)
        warning(message('nnet_cnn_onnx:onnx:PReluSlopeDim',LayerName));
        NNTLayer = [];
        return;
    end
    numChannels   = numel(slope);
    DLTChannelDim = 3;  % [H W C N]
end
NNTLayer = nnet.onnx.layer.PreluLayer(LayerName, numChannels, DLTChannelDim, slope);
end

function tf = RNNSlopeDimOK(slopeDim)
% Supported slopeDim can be [C] or [1 C] or [1 1 C]
tf = numel(slopeDim)==1 || all(slopeDim(1:numel(slopeDim)-1)==1);
end

function tf = CNNSlopeDimOK(slopeDim)
% Supported slopeDim can be [C 1 1] or [1 C 1 1]
tf = numel(slopeDim)==3 && isequal(slopeDim([2 3]), [1 1]) || ...
    numel(slopeDim)==4 && isequal(slopeDim([1 3 4]), [1 1 1]);
end