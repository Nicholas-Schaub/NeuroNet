function NNTLayer = translateImageScaler(node, LayerName, OpsetVersion)
%https://github.com/onnx/onnx/blob/master/docs/Operators.md#ImageScaler
% Bias : list of floats. Bias applied to each channel, same size as C.
% Scale : float (scalar)

%   Copyright 2018 The MathWorks, Inc.
attributeNames = arrayfun(@(a) a.name, node.attribute,'UniformOutput',false);
bias = single(1);
scale = single(1);
for t =1: numel(attributeNames)
    switch attributeNames{t}
        case 'bias'
            bias  = single(node.attribute(t).floats);
        case 'scale'
            scale =  single(node.attribute(t).f);
    end
end
NNTLayer = nnet.onnx.layer.ElementwiseAffineLayer(LayerName, scale, reshape(bias, [1 1 numel(bias)]));
end