classdef ConverterForCrossChannelNormalizationLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a CrossChannelNormalizationLayer into ONNX
        
    % Copyright 2018 The Mathworks, Inc.
    
    methods
        function this = ConverterForCrossChannelNormalizationLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, parameterInputs, networkInputs, networkOutputs, layerMap] = toOnnx(this, TensorNameMap)
            import nnet.internal.cnn.onnx.*

            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            nodeProto = NodeProto;
            nodeProto.op_type   = 'LRN';
            nodeProto.name      = onnxName;
            nodeProto.input     = mapTensorNames(this, this.InputLayerNames(1), TensorNameMap);
            nodeProto.output    = {onnxName};
            
            nodeProto.attribute = [...
                makeAttributeProto('alpha',	'FLOAT', this.NNTLayer.Alpha),...
                makeAttributeProto('beta',	'FLOAT', this.NNTLayer.Beta),...
                makeAttributeProto('bias',	'FLOAT', this.NNTLayer.K),...
                makeAttributeProto('size',	'INT',   this.NNTLayer.WindowChannelSize)
                ];
            
            parameterInitializers   = [];
            parameterInputs         = [];
            networkInputs       	= [];
            networkOutputs          = [];
            layerMap                = containers.Map;
            
            if nameChanged
                layerMap(this.NNTLayer.Name) = onnxName;
            end
        end
    end
end
