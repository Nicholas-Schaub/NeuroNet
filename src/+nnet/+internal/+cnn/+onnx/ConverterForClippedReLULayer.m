classdef ConverterForClippedReLULayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert an clippedReluLayer into ONNX
        
    % Copyright 2018 The Mathworks, Inc.
    
    methods
        function this = ConverterForClippedReLULayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, parameterInputs, networkInputs, networkOutputs, layerMap] = toOnnx(this, TensorNameMap)
            % An NNT clippedReluLayer translates into a Relu followed by a Clip.
            import nnet.internal.cnn.onnx.*
            
            [ONNXName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            % (1) Relu: Takes input X
            ReluNodeName            = [ONNXName '_Relu'];
            nodeProto(1)            = NodeProto;
            nodeProto(1).op_type    = 'Relu';
            nodeProto(1).name     	= ReluNodeName;
            nodeProto(1).input      = mapTensorNames(this, this.InputLayerNames(1), TensorNameMap);
            nodeProto(1).output    = {ReluNodeName};
            
            % (2) Clip: Takes input ReluNodeName
            ClipNodeName        	= [ONNXName '_Clip'];
            nodeProto(2)            = NodeProto;
            nodeProto(2).op_type    = 'Clip';
            nodeProto(2).name       = ClipNodeName;
            nodeProto(2).input      = {ReluNodeName};
            nodeProto(2).output    = {ClipNodeName};
            % Clip Attributes
            nodeProto(2).attribute = [...
                makeAttributeProto('min', 'FLOAT', 0),...
                makeAttributeProto('max', 'FLOAT', this.NNTLayer.Ceiling)
                ];
            
            parameterInitializers   = [];
            parameterInputs         = [];
            networkInputs           = [];
            networkOutputs          = [];
            layerMap                = containers.Map;
            
            % Set the output tensor name to the Clip name
            layerMap(this.NNTLayer.Name) = ClipNodeName;
        end
    end
end
