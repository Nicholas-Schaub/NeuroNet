classdef ConverterForClassificationOutputLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a ClassificationOutputLayer into ONNX
        
    % Copyright 2018-2019 The Mathworks, Inc.
    
    methods
        function this = ConverterForClassificationOutputLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, parameterInputs, networkInputs, networkOutputs, layerMap] = toOnnx(this, TensorNameMap)
            import nnet.internal.cnn.onnx.*

            inputLayerName          = mapTensorNames(this, this.InputLayerNames(1), TensorNameMap);
            
            nodeProto               = [];
            parameterInitializers   = [];
            parameterInputs         = [];
            networkInputs           = [];
            assert(isscalar(this.NNTLayer.OutputSize));
            networkOutputs          = makeValueInfoProtoFromDimensions(...
                inputLayerName{1}, ...
                TensorProto_DataType.FLOAT, ...
                [1 this.NNTLayer.OutputSize]);	% NNT output size is a scalar, ONNX tensor is [n c] after softmax at this point.
            layerMap                = containers.Map;
        end
    end
end
