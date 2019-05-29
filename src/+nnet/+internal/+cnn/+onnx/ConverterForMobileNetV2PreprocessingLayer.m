classdef ConverterForMobileNetV2PreprocessingLayer < nnet.internal.cnn.onnx.ConverterForAffineLayersBaseClass
    % Class to convert a nnet.mobilenetv2.layer.MobileNetV2PreprocessingLayer into ONNX.
    % Y=(1/128)X-1
        
    % Copyright 2019 The Mathworks, Inc.
    
    methods
        function this = ConverterForMobileNetV2PreprocessingLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.ConverterForAffineLayersBaseClass(layerAnalyzer);
            this.a = 1/128;
            this.b = -1;
        end
    end
end
