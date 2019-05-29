function NNTLayer = translateMul(node, LayerName, initializerDimMap, initializerRawDataMap, initializerNames, OpsetVersion)

%   Copyright 2018 The MathWorks, Inc.

% Must have exactly 2 inputs
if numel(node.input) ~= 2
    warning(message('nnet_cnn_onnx:onnx:NumNodeInputs', 'Mul', 2, LayerName));
    NNTLayer = [];
    return;
end
isParam = cellfun(@(a)ismember(a, initializerNames),{node.input},'UniformOutput',false);
isParam = isParam{:};
% One input must be a constant.
if sum(isParam) ~= 1
    warning(message('nnet_cnn_onnx:onnx:OneNodeOneConst', 'Mul', LayerName));
    NNTLayer = [];
    return;
end
% Make a ScalingFactorLayer.
paramIdx = find(isParam,1);
constName = node.input(paramIdx);
constName = constName{:};
constDim  = CalcConstDim(node, OpsetVersion, constName, initializerDimMap);
% Convert the const from FCHW row-major to HWCF col-major:
const     = single(initializerRawDataMap(constName));
const     = reshape(const, fliplr(constDim));                      % Now it's WHCN col-maj.
const     = permute(const, [2 1 3 4]);                            % Now it's HWCN col-maj.
% Create layer
NNTLayer = nnet.onnx.layer.ElementwiseAffineLayer(LayerName, const, 0);
end

function constDim = CalcConstDim(node, OpsetVersion, constName, initializerDimMap)
if OpsetVersion <= 6
    % Get the attributes
    broadcast   = [];
    axis        = [];
    attributeNames = arrayfun(@(a) a.name, node.attribute,'UniformOutput',false);
    for t = 1:numel(attributeNames)
        switch attributeNames{t}
            case 'broadcast'
                broadcast = node.attribute(t).i;
            case 'axis'
                axis = node.attribute(t).i;
        end
    end
    constDim = initializerDimMap(constName);
    if broadcast
        % Broadcast the const to 4D
        if ~isempty(axis)
            constDim = [ones(1,axis), constDim];                  % Prepend 'axis' ones.
            constDim = [constDim ones(1,4-numel(constDim))];       % Postpend ones to 4D.
        else
            constDim = [ones(1,4-numel(constDim)), constDim];      % Default axis: Prepend ones to 4D.
        end
    else
        constDim = [constDim ones(1,4-numel(constDim))];       % Postpend ones to 4D.
    end
else
    % Opset >= 7
    constDim = nnet.internal.cnn.onnx.multidirectionalBroadcast(initializerDimMap(constName));
end
end
