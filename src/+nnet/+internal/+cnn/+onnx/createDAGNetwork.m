function NNTNetwork = createDAGNetwork(LayerGraph,Classes)
% Create a DAGNetwork without training.

% Copyright 2018 The Mathworks, Inc.

C = suppressAutoClassesWarning();
if any(arrayfun(@isUnsupportedLayer, LayerGraph.Layers))
    throwAsCaller(MException(message('nnet_cnn_onnx:onnx:CantBuildNetWithUnsupportedLayers')));
end

if ~iIsAuto(Classes) && any(arrayfun(@(layer)isa(layer,'nnet.cnn.layer.ClassificationOutputLayer') && iIsAuto(layer.Classes), ...
                                     LayerGraph.Layers))
    % Assemble the network to determine how many classes are required in its
    % output layer.
    NNTNetwork = assembleNetwork(LayerGraph);
    LayerGraph = layerGraph(NNTNetwork);
end
lg = LayerGraph;
for i = 1:numel(lg.Layers)
    layer = lg.Layers(i);
    if isa(layer,'nnet.cnn.layer.ClassificationOutputLayer')
        if iIsAuto(Classes)
            % assembleNetwork will provide default class names
            warning(message('nnet_cnn_onnx:onnx:FillingInClassNames'));
        else
            iVerifyNumClasses(layer.Classes, Classes);
            if iscategorical(Classes)
                classes = Classes;          % This preserves the ordinal property.
            else
                classes = categorical(Classes, Classes);
            end
            layer.Classes = classes;
            lg = replaceLayer(lg, layer.Name, layer);
        end
    elseif isa(layer,'nnet.cnn.layer.RegressionOutputLayer')
        %Current DAGs only support one output layer. We error out if class
        %names are provided for regression output layer.
        if ~iIsAuto(Classes)
            throwAsCaller(MException(message('nnet_cnn_onnx:onnx:ClassNamesForRegression')));
        end
    end
end
NNTNetwork = assembleNetwork(lg);
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
