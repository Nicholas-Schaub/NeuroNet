classdef ConverterForPixelClassificationLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a PixelClassificationLayer into ONNX
        
    % Copyright 2018 The Mathworks, Inc.

    methods
        function this = ConverterForPixelClassificationLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, parameterInputs, networkInputs, networkOutputs, layerMap] = toOnnx(this, TensorNameMap)
            import nnet.internal.cnn.onnx.*

            inputLayerName          = mapTensorNames(this, this.InputLayerNames(1), TensorNameMap);
            nodeProto               = [];
            parameterInitializers   = [];
            parameterInputs         = [];
            networkInputs           = [];
            assert(numel(this.NNTLayer.OutputSize)==3);
            h = this.NNTLayer.OutputSize(1);
            w = this.NNTLayer.OutputSize(2);
            c = this.NNTLayer.OutputSize(3);
            networkOutputs          = makeValueInfoProtoFromDimensions(...
                inputLayerName{1}, ...
                TensorProto_DataType.FLOAT, ...
                [1 c h w]);                                                 % NNT output size is hwc, ONNX is nchw
            layerMap                = containers.Map;
        end
    end
end
