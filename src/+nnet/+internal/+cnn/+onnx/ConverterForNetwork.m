classdef ConverterForNetwork
    % Class to convert a Network into ONNX
    
    %  Example ModelProto:
    %           ir_version: 1
    %         opset_import: []
    %        producer_name: 'onnx-caffe2'
    %     producer_version: []
    %               domain: []
    %        model_version: []
    %           doc_string: []
    %                graph: [1×1 nnet.internal.cnn.onnx.GraphProto]
    %       metadata_props: []
    
    % Copyright 2018 The Mathworks, Inc.
    
    
    properties
        Network
        Metadata
    end
    
    methods
        function this = ConverterForNetwork(network, metadata)
            this.Network = network;
            this.Metadata = metadata;
        end
        
        function modelProto = toOnnx(this)
            import nnet.internal.cnn.onnx.*
            % Set ONNX operatorSet version number
            opsetIdProto = OperatorSetIdProto;
            opsetIdProto.version = int64(this.Metadata.OpsetVersion);
            
            % ModelProto fields
            modelProto                  = ModelProto;
            modelProto.ir_version       = int64(this.Metadata.IrVersion);
            modelProto.opset_import     = opsetIdProto;
            modelProto.producer_name    = char(this.Metadata.ProducerName);
            modelProto.producer_version = char(this.Metadata.ProducerVersion);
            modelProto.domain           = char(this.Metadata.Domain);
            modelProto.model_version  	= int64(this.Metadata.ModelVersion);
            modelProto.doc_string       = char(this.Metadata.DocString);
            modelProto.graph            = networkToGraphProto(this);
            modelProto.metadata_props	= [];
            
            % Optionally add MathWorks operatorSet
            if MathWorksOperatorsUsed(modelProto)
                opsetIdProto = OperatorSetIdProto;
                opsetIdProto.domain = 'com.mathworks';
                opsetIdProto.version = int64(this.Metadata.MathWorksOpsetVersion);
                modelProto.opset_import(end+1) = opsetIdProto;
            end
        end
        
        function graphProto = networkToGraphProto(this)
            %   Example GraphProto:
            %            node: [1×66 nnet.internal.cnn.onnx.NodeProto]
            %            name: 'squeezenet'
            %     initializer: [1×52 nnet.internal.cnn.onnx.TensorProto]
            %      doc_string: []
            %           input: [1×53 nnet.internal.cnn.onnx.ValueInfoProto]
            %          output: [1×1 nnet.internal.cnn.onnx.ValueInfoProto]
            %      value_info: []
            import nnet.internal.cnn.onnx.*
            
            % Convert layers while gathering nodes, initializers, network
            % inputs, network outputs, and the TensorNameMap.
            nodeProtos              = [];
            parameterInitializers   = [];
            parameterInputs         = [];
            networkInputs           = [];
            networkOutputs          = [];
            TensorNameMap           = containers.Map;
            networkAnalysis         = nnet.internal.cnn.analyzer.NetworkAnalyzer(this.Network);
            IsRecurrentNetwork      = isa(this.Network.Layers(1), 'nnet.cnn.layer.SequenceInputLayer');
            for layerNum = 1:numel(networkAnalysis.LayerAnalyzers)
                % LayerAnalyzers are in topological order
                layerAnalyzer           = networkAnalysis.LayerAnalyzers(layerNum);
                layerConverter          = NNTLayerConverter.makeLayerConverter(layerAnalyzer, this.Metadata.OpsetVersion, IsRecurrentNetwork);
                [nodeProto, paramInitializers, paramInputs, netInputs, netOutputs, layerMap] = toOnnx(layerConverter, TensorNameMap);
                nodeProtos              = [nodeProtos, nodeProto];
                parameterInitializers   = [parameterInitializers, paramInitializers];
                parameterInputs     	= [parameterInputs, paramInputs];
                networkInputs           = [networkInputs, netInputs];
                networkOutputs          = [networkOutputs, netOutputs];
                TensorNameMap           = [TensorNameMap; layerMap];    % Note vertical cat
            end
            
            % Set graphProto fields
            graphProto              = GraphProto;
            graphProto.name         = this.Metadata.NetworkName;
            graphProto.node         = nodeProtos;
            graphProto.initializer	= parameterInitializers;
            graphProto.input        = [parameterInputs, networkInputs];
            graphProto.output       = networkOutputs;
        end
    end
end


function tf = MathWorksOperatorsUsed(modelProto)
nodes = modelProto.graph.node;
tf = any(arrayfun(@(n)isequal(lower(n.domain), 'com.mathworks'), nodes));
end

% function names = findLayerInputNames(network, layerNum)
% % Names of layers that directly input to layerNum. Does not use
% % NetworkAnalyzer
% if isa(network, 'SeriesNetwork')
%     if layerNum == 1
%         names = {};
%     else
%         names = {network.Layers(layerNum-1)};
%     end
% else
%     assert(isa(network, 'DAGNetwork'));
%     lg = layerGraph(network);
%     en = lg.HiddenConnections.EndNodes;
%     inconnLayerNums = en(en(:,2) == layerNum);
%     names = arrayfun(@(L)L.Name, network.Layers(inconnLayerNums), 'UniformOutput', false);
% end
% end
