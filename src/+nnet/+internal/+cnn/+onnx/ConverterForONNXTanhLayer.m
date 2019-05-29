classdef ConverterForONNXTanhLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a nnet.onnx.layer.TanhLayer into ONNX
    
    % Copyright 2018 The Mathworks, Inc.
    
    methods
        function this = ConverterForONNXTanhLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, parameterInputs, networkInputs, networkOutputs, layerMap] = toOnnx(this, TensorNameMap)
            import nnet.internal.cnn.onnx.*
            
            [NodeName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            nodeProto          	= NodeProto;
            nodeProto.op_type	= 'Tanh';
            nodeProto.name     	= NodeName;
            nodeProto.input   	= mapTensorNames(this, this.InputLayerNames(1), TensorNameMap);
            nodeProto.output  	= {NodeName};
            
            parameterInitializers   = [];
            parameterInputs         = [];
            networkInputs           = [];
            networkOutputs          = [];
            layerMap                = containers.Map;
            
            % Maybe update the layer name map
            if nameChanged
                layerMap(this.NNTLayer.Name) = NodeName;
            end
        end
    end
end

