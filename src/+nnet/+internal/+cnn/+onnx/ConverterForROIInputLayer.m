classdef ConverterForROIInputLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a ROIInputLayer into ONNX
    
    % Copyright 2018 The Mathworks, Inc.
    
    methods
        function this = ConverterForROIInputLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, parameterInputs, networkInputs, networkOutputs, layerMap] = toOnnx(this, TensorNameMap)
            import nnet.internal.cnn.onnx.*
            % Generate a ValueInfoProto describing the input tensor.
            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            outputSize      = this.OutputSize{1};
            
            inputTensorSize = [outputSize(1), 5];                % NNT is size [numROI 4], ONNX is size [numROI 5]
            networkInputs  	= makeValueInfoProtoFromDimensions(onnxName, TensorProto_DataType.FLOAT, inputTensorSize);  
            layerMap       	= containers.Map;
            networkOutputs	= [];
            nodeProto       = [];
            parameterInitializers = [];
            parameterInputs = [];
            
            if nameChanged
                layerMap(name) = onnxName;
            end
        end
    end
end
