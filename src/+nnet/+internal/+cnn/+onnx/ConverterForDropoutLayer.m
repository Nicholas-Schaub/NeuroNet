classdef ConverterForDropoutLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a DropoutLayer into ONNX
    
    % Copyright 2018 The Mathworks, Inc.
    
    methods
        function this = ConverterForDropoutLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, parameterInputs, networkInputs, networkOutputs, layerMap] = toOnnx(this, TensorNameMap)
            import nnet.internal.cnn.onnx.*
            
            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            % Make the nodeProto
            nodeProto           = NodeProto;
            nodeProto.op_type	= 'Dropout';
            nodeProto.name      = onnxName;
            nodeProto.input     = mapTensorNames(this, this.InputLayerNames(1), TensorNameMap);
            nodeProto.output    = {onnxName};
            nodeProto.attribute = makeAttributeProto('ratio', 'FLOAT', this.NNTLayer.Probability);
            if this.OpsetVersion < 7
                nodeProto.attribute(1,2) = makeAttributeProto('is_test', 'INT', 1);
            end
            
            parameterInitializers   = [];
            parameterInputs         = [];
            networkInputs           = [];
            networkOutputs          = [];
            layerMap                = containers.Map;
            
            if nameChanged
                layerMap(this.NNTLayer.Name) = onnxName;
            end
        end
    end
end

