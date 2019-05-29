classdef ConverterForAdditionLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert an additionLayer into ONNX
        
    % Copyright 2018 The Mathworks, Inc.
    
    methods
        function this = ConverterForAdditionLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, parameterInputs, networkInputs, networkOutputs, layerMap] = toOnnx(this, TensorNameMap)
            import nnet.internal.cnn.onnx.*
            
            [ONNXName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            nodeProto           = NodeProto;
            nodeProto.op_type   = 'Sum';
            nodeProto.name      = ONNXName;
            nodeProto.input     = mapTensorNames(this, this.InputLayerNames(:)', TensorNameMap);
            nodeProto.output{1} = ONNXName;
            
            parameterInitializers   = [];
            parameterInputs         = [];
            networkInputs           = [];
            networkOutputs          = [];
            layerMap                = containers.Map;
            
            if nameChanged
                layerMap(this.NNTLayer.Name) = ONNXName;
            end
        end
    end
end
