classdef ConverterForInceptionResNetV2ScalingFactorLayer < nnet.internal.cnn.onnx.ConverterForAffineLayersBaseClass
    % Class to convert a nnet.inceptionresnetv2.layer.ScalingLayer into ONNX.
    % Y=(2/255)X-1
        
    % Copyright 2018 The Mathworks, Inc.
    
    methods
        function this = ConverterForInceptionResNetV2ScalingFactorLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.ConverterForAffineLayersBaseClass(layerAnalyzer);
            this.a = this.NNTLayer.Scale;
            this.b = 0;
        end
    end
end
