classdef ConverterForInceptionV3ScalingLayer < nnet.internal.cnn.onnx.ConverterForAffineLayersBaseClass
    % Class to convert a nnet.inceptionv3.layer.ScalingLayer into ONNX.
    % Y=(2/255)X-1
        
    % Copyright 2018 The Mathworks, Inc.
    
    methods
        function this = ConverterForInceptionV3ScalingLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.ConverterForAffineLayersBaseClass(layerAnalyzer);
            this.a = 2/255;
            this.b = -1;
        end
    end
end
