function NNTLayer = translateAdd(node, LayerName, initializerDimMap, initializerRawDataMap, initializerNames, OpsetVersion)

%   Copyright 2018 The MathWorks, Inc.
isParam = cellfun(@(a)ismember(a, initializerNames),{node.input},'UniformOutput',false);
isParam = isParam{:};
if sum(isParam) == 0
    % All inputs are other layers. Make an additionLayer.
    NNTLayer = additionLayer(numel(node.input), 'Name', LayerName);
else
    % One input is a constant. Make a BiasLayer.
    paramIdx = find(isParam,1);
    constName = node.input(paramIdx);
    constName = constName{:};
    constDim  = CalcConstDim(node, OpsetVersion, constName, initializerDimMap);  % constDim is ONNX shape: NCHW
    % Convert the const from NCHW row-major to HWCN col-major:
    const     = single(initializerRawDataMap(constName));   % It's NCHW rowmaj.
    const     = reshape(const, fliplr(constDim));         	% Now it's WHCN colmaj.
    const     = permute(const, [2 1 3 4]);                	% Now it's HWCN colmaj.
    % Create layer
    NNTLayer = nnet.onnx.layer.ElementwiseAffineLayer(LayerName, 1, const);
end
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
