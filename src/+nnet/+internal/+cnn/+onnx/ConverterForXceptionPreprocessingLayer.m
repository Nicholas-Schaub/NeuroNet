classdef ConverterForXceptionPreprocessingLayer < nnet.internal.cnn.onnx.ConverterForAffineLayersBaseClass
    % Class to convert a nnet.xception.layer.XceptionPreprocessingLayer into ONNX.
    % Y=(1/127.5)X-1
        
    % Copyright 2019 The Mathworks, Inc.
    
    methods
        function this = ConverterForXceptionPreprocessingLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.ConverterForAffineLayersBaseClass(layerAnalyzer);
            this.a = 1/127.5;
            this.b = -1;
        end
    end
end
