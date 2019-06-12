classdef ConverterForNASNetMobilePreprocessingLayer < nnet.internal.cnn.onnx.ConverterForAffineLayersBaseClass
    % Class to convert a nnet.nasnetmobile.layer.NASNetMobilePreprocessingLayer into ONNX.
    % Y=(2/255)X-1
        
    % Copyright 2019 The Mathworks, Inc.
    
    methods
        function this = ConverterForNASNetMobilePreprocessingLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.ConverterForAffineLayersBaseClass(layerAnalyzer);
            this.a = 2/255;
            this.b = -1;
        end
    end
end
