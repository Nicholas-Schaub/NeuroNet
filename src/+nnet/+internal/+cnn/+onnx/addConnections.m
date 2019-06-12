function NNTLayerGraph = addConnections(NNTLayerGraph, graph, destinationLayerNameMap, initializerNames)
% initializerNames = arrayfun(@(a) a.name, graph.initializer,'UniformOutput',false);

%create a cell array of input list
allInputs = cell(numel(graph.node),0);
for i=1:numel(graph.node)
    allInputs{i} = graph.node(i).input;
end

if numel(graph.output) > 1
    error(message('nnet_cnn_onnx:onnx:MO'));
end

isParam=cellfun(@(a)ismember(a, initializerNames),allInputs,'UniformOutput',false);
for i=1:numel(allInputs)
    allInputs{i}(isParam{i})=[]; %remove all initializers from the input lists, leaving only nodes and graph input.
end

for i=1:numel(allInputs)
    if numel(allInputs{i}) == 1     % Node with one input
        NNTTargetString =  graph.node(i).name;
        if isempty(NNTTargetString)
            NNTTargetString =  strcat("node_",string(i));
        end
        if isKey(destinationLayerNameMap, char(NNTTargetString))
            NNTTargetString = string(destinationLayerNameMap(char(NNTTargetString)));
        end
        NNTSourceString = sourceTensorString(graph, allInputs{i}, NNTLayerGraph);
        NNTLayerGraph = connectLayers(NNTLayerGraph, NNTSourceString, NNTTargetString);
    else    % node with multiple inputs
        NNTTargetLayerName = graph.node(i).name;
        if isempty(NNTTargetLayerName)
            NNTTargetLayerName =  strcat("node_",string(i));
        end
        NNTTargetLayer = NNTLayerFromName(NNTLayerGraph, NNTTargetLayerName);
        for j= 1:numel(allInputs{i})
            NNTTargetString = sprintf('%s/%s', NNTTargetLayerName, NNTTargetLayer.InputNames{j});
            NNTSourceString = sourceTensorString(graph, allInputs{i}{j}, NNTLayerGraph);
            NNTLayerGraph = connectLayers(NNTLayerGraph, NNTSourceString, NNTTargetString);
        end
    end
end
end

function NNTSourceString = sourceTensorString(graph, outputTensorName, NNTLayerGraph)
% (1) Find source node and output tensor index
nodes = graph.node;
for sourceNodeIdx = 1:numel(nodes)
    sourceNode = graph.node(sourceNodeIdx);
    [found, tensorOutputIdx] = ismember(outputTensorName, sourceNode.output);
    if found
        break;
    end
end
% (2) Find or build the source tensor name
if ~found   % Input connection source not found. Source is the input layer.
    NNTSourceString = NNTLayerGraph.Layers(1).Name;
else    % Source is not the input layer
    sourceNodeName = sourceNode.name;
    if isempty(sourceNodeName)
        sourceNodeName = strcat("node_",string(sourceNodeIdx));
    end
    if numel(sourceNode.output) > 1     % ONNX source node is multi-output
        NNTSourceLayer = NNTLayerFromName(NNTLayerGraph, sourceNodeName);
        NNTSourceTensorName = NNTSourceLayer.OutputNames{tensorOutputIdx};
        NNTSourceString = sprintf('%s/%s', sourceNodeName, NNTSourceTensorName);
    else    % Source node is single-output
        NNTSourceString = sourceNodeName;
    end
end
end

function layer = NNTLayerFromName(lg, name)
Layers = lg.Layers;
for i=1:numel(Layers)
    if isequal(Layers(i).Name, name)
        layer = Layers(i);
        return;
    end
end
assert(false);
end