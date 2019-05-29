function NNTLayer = translateMaxPool(thisNode, LayerName, OpsetVersion)
%https://github.com/onnx/onnx/blob/master/docs/Operators.md#MaxPool

attributeNames = arrayfun(@(a) a.name, thisNode.attribute,'UniformOutput',false);
attributeInts = arrayfun(@(a) a.ints, thisNode.attribute,'UniformOutput',false);
attributeIntsMap = containers.Map(attributeNames, attributeInts);
attributeInt = arrayfun(@(a) a.i, thisNode.attribute,'UniformOutput',false);
attributeIntMap = containers.Map(attributeNames, attributeInt);
attributeString = arrayfun(@(a) a.s, thisNode.attribute,'UniformOutput',false);
attributeStringMap = containers.Map(attributeNames, attributeString);
% kernel_shape
FilterSize = double(attributeIntsMap('kernel_shape'));

% Handle padding
Padding = [0 0 0 0];
if ismember('auto_pad',attributeNames)
    auto_pad = attributeStringMap('auto_pad');
    switch auto_pad
        case 'SAME_UPPER'
            Padding = 'same';
        case 'SAME_LOWER'
            Padding = 'same';
            warning(message('nnet_cnn_onnx:onnx:AutoPadSameLower',LayerName));
        case 'VALID'
            Padding = [0 0 0 0];
    end
elseif ismember('pads', attributeNames)
    Padding = double(attributeIntsMap('pads'));
    %ONNX: [H_b,W_b,H_end,W_end] ==> [t l b r]
    %MATLAB: [t b l r]
    Padding = Padding([1,3,2,4]);
end

% storage_order
if ismember('storage_order', attributeNames)
    storage_order = double(attributeIntMap('storage_order'));
    if storage_order==1
        error(message('nnet_cnn_onnx:onnx:MaxPoolStorageOrder', LayerName));
    end
end
% strides
if ismember('strides', attributeNames)
    Stride =  double(attributeIntsMap('strides')); %[h w]
else
    Stride = [1 1];
end
if OpsetVersion >= 8
    % storage_order
    if ismember('storage_order', attributeNames)
        storage_order = double(attributeIntMap('storage_order'));
        if storage_order==1
            % column-major not supported
            warning(message('nnet_cnn_onnx:onnx:MaxPoolStorageOrder', LayerName));
            NNTLayer = [];
            return;
        end
    end
end

NNTLayer = maxPooling2dLayer(FilterSize, 'Stride', Stride, 'Padding', Padding, 'Name', LayerName);
end