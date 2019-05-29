classdef ConverterForResNet18PreprocessingLayer < nnet.internal.cnn.onnx.ConverterForImageStandardizationLayersBaseClass
    % Class to convert a nnet.resnet18.layer.ResNet18PreprocessingLayer into ONNX.
    % Y = (X/255 - meanC)./stdC, where X is [H W C N], and meanC and stdC
    % are Cx1.
        
    % Copyright 2018 The Mathworks, Inc.
    
    methods
        function this = ConverterForResNet18PreprocessingLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.ConverterForImageStandardizationLayersBaseClass(layerAnalyzer);
            meanC = [0.485, 0.456, 0.406]';
            stdC  = [0.229, 0.224, 0.225]';
            this.Scale = 1./(255*stdC);
            this.Offset = -meanC./stdC;
        end
    end
end
