classdef ConverterForRCNNBoxRegressionLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a RCNNBoxRegressionLayer into ONNX
        
    % Copyright 2018 The Mathworks, Inc.
    
    properties
        LayerAnalyzer
    end

    methods
        function this = ConverterForRCNNBoxRegressionLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
            this.LayerAnalyzer = layerAnalyzer;
        end
        
        function [nodeProto, parameterInitializers, parameterInputs, networkInputs, networkOutputs, layerMap] = toOnnx(this, TensorNameMap)
            import nnet.internal.cnn.onnx.*

            inputLayerName          = mapTensorNames(this, this.InputLayerNames(1), TensorNameMap);
            nodeProto               = [];
            parameterInitializers   = [];
            parameterInputs         = [];
            networkInputs           = [];
            hwcSize                 = this.LayerAnalyzer.Inputs.Size{1}(:)';
            networkOutputs          = makeValueInfoProtoFromDimensions(...
                inputLayerName{1}, ...
                TensorProto_DataType.FLOAT, ...
                [1 hwcSize([3 1 2])]);	% NNT output size is hwcn, ONNX is nchw with n=1
            layerMap                = containers.Map;
        end
    end
end
