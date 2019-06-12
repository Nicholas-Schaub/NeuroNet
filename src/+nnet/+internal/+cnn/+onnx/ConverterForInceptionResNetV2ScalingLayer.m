classdef ConverterForInceptionResNetV2ScalingLayer < nnet.internal.cnn.onnx.ConverterForAffineLayersBaseClass
    % Class to convert a nnet.inceptionresnetv2.layer.ScalingLayer into ONNX.
    % Y=(2/255)X-1
        
    % Copyright 2018 The Mathworks, Inc.
    
    methods
        function this = ConverterForInceptionResNetV2ScalingLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.ConverterForAffineLayersBaseClass(layerAnalyzer);
            this.a = 2/255;
            this.b = -1;
        end
    end
end
