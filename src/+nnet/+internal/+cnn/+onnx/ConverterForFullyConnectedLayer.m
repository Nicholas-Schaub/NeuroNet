classdef ConverterForFullyConnectedLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a FullyConnectedLayer into ONNX
    
    % Copyright 2018-2019 The Mathworks, Inc.
    
    properties
        LayerAnalyzer
    end
    
    methods
        function this = ConverterForFullyConnectedLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
            this.LayerAnalyzer = layerAnalyzer;
        end
        
        function [nodeProto, parameterInitializers, parameterInputs, networkInputs, networkOutputs, layerMap] = toOnnx(this, TensorNameMap)
            % An NNT FullyConnectedLayer translates into a Flatten followed by a Gemm.
            import nnet.internal.cnn.onnx.*
            
            [gemmName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            WName = [gemmName '_W'];
            BName = [gemmName '_B'];
            
            % Determine whether to flatten the input tensor 
            inputSize = this.InputLayerSizes{1};
            flattenInputTensor = numel(inputSize)>1;
            
            % (1) Flatten (if needed)
            if flattenInputTensor
                FlattenNodeName        = [gemmName '_Flatten'];
                FlattenNode            = NodeProto;
                FlattenNode.op_type    = 'Flatten';
                FlattenNode.name       = FlattenNodeName;
                FlattenNode.input      = mapTensorNames(this, {this.InputLayerNames{1}}, TensorNameMap);
                FlattenNode.output     = {FlattenNodeName};
                FlattenNode.attribute  = makeAttributeProto('axis', 'INT', 1);
                GemmNodeInput          = {FlattenNodeName, WName, BName};
            else
                FlattenNode     = [];
                GemmNodeInput   = mapTensorNames(this, {this.InputLayerNames{1}, WName, BName}, TensorNameMap);
            end
            
            % (2) Gemm
            GemmNode           = NodeProto;
            GemmNode.op_type   = 'Gemm';
            GemmNode.name      = gemmName;
            GemmNode.input     = GemmNodeInput;
            GemmNode.output    = {gemmName};
            GemmNode.attribute = [...
                makeAttributeProto('alpha',     'FLOAT', 1),...
                makeAttributeProto('beta',      'FLOAT', 1),...
                makeAttributeProto('transA',  	'INT',   0),...
                makeAttributeProto('transB',   	'INT',   1)
                ];
            if this.OpsetVersion < 7
                GemmNode.attribute(end+1) = makeAttributeProto('broadcast', 'INT', 1);
            end
            
            nodeProto = [FlattenNode, GemmNode];    % Note: FlattenNode may be [].
            
            % Make parameter Initializers
            t1              = TensorProto;
            t1.name         = WName;             % W = Weights
            t1.data_type	= TensorProto_DataType.FLOAT;
            if this.IsRecurrentNetwork
                % Weights are stored in the external layer as FH, where H is numHidden from the previous layer.
                t1.raw_data     = rawData(single(this.NNTLayer.Weights));                               % Convert from col-maj to row-maj and store it.
                t1.dims         = dimVector(size(this.NNTLayer.Weights),2);                             % Store dims as [F H]. Remember we're then telling it to transpose this and right-multiply X by it.
            else
                % Weights are stored in the InternalLayer as HWCF row-major.
                W               = permute(this.LayerAnalyzer.InternalLayer.Weights.Value, [4 3 1 2]);	% Convert from HWCF to FCHW
                t1.raw_data     = rawData(single(W));                                                   % Convert from col-maj to row-maj while still 4D, and store it.
                t1.dims         = dimVector(size(this.NNTLayer.Weights),2);                             % Then declare the dimensions to be 2D: [F, CHW]. Remember we're then telling it to transpose this and right-multiply X by it.
            end
            
            t2              = TensorProto;
            t2.name         = BName;                                    % B = Bias
            t2.data_type	= TensorProto_DataType.FLOAT;
            t2.raw_data     = rawData(single(this.NNTLayer.Bias));
            t2.dims         = dimVector(numel(this.NNTLayer.Bias),1);   % This is 1D: [F].
            
            parameterInitializers = [t1 t2];
            
            % Make parameter Inputs
            parameterInputs = arrayfun(@makeValueInfoProtoFromTensorProto, parameterInitializers);
            
            networkInputs           = [];
            networkOutputs          = [];
            layerMap                = containers.Map;
            
            if nameChanged
                % Set the output tensor name
                layerMap(this.NNTLayer.Name) = gemmName;
            end
        end
    end
end