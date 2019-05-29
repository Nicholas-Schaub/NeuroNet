function NNTNetwork = createSeriesNetwork(LayerArray, Classes)
% Create a SeriesNetwork without training.

% Copyright 2017-2018 The MathWorks, Inc.

C = suppressAutoClassesWarning();
if any(arrayfun(@isUnsupportedLayer, LayerArray))
    throwAsCaller(MException(message('nnet_cnn_onnx:onnx:CantBuildNetWithUnsupportedLayers')));
end

if isa(LayerArray(end),'nnet.cnn.layer.ClassificationOutputLayer')
    if iIsAuto(Classes)
        % assembleNetwork will provide default class names
        warning(message('nnet_cnn_onnx:onnx:FillingInClassNames'));
    else
        % Classes were passed
        if iIsAuto(LayerArray(end).Classes)
            % Assemble the network to determine how many classes are required in its
            % output layer.
            NNTNetwork = assembleNetwork(LayerArray);
            LayerArray = NNTNetwork.Layers;
        end
        % Last layer now has Classes
        iVerifyNumClasses(LayerArray(end).Classes, Classes);
        if iscategorical(Classes)
            classes = Classes;          % This preserves the ordinal property.
        else
            classes = categorical(Classes, Classes);
        end
        LayerArray(end).Classes = classes;
    end
elseif isa(LayerArray(end),'nnet.cnn.layer.RegressionOutputLayer')
    if ~iIsAuto(Classes)
        % We error out if class names are provided for regression output layer.
        throwAsCaller(MException(message('nnet_cnn_onnx:onnx:ClassNamesForRegression')));
    end
end
NNTNetwork = assembleNetwork(LayerArray);
end

function tf = isUnsupportedLayer(Layer)
tf = isa(Layer, 'nnet.onnx.layer.PlaceholderLayer');
end

function tf = iIsAuto(val)
tf = isequal(string(val), "auto");
end

function C = suppressAutoClassesWarning()
warnState = warning('off', 'nnet_cnn:internal:cnn:analyzer:NetworkAnalyzer:NetworkHasWarnings');
C = onCleanup(@()warning(warnState));
end

function iVerifyNumClasses(NetClasses, PassedClasses)
if numel(NetClasses) ~= numel(PassedClasses)
    throwAsCaller(MException(message('nnet_cnn_onnx:onnx:ClassNamesMismatchSize', numel(NetClasses), numel(PassedClasses))));
end
end
