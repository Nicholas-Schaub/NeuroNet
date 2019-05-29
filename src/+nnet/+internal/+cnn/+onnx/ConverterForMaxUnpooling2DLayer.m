classdef ConverterForMaxUnpooling2DLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a MaxUnpooling2dLayer into ONNX
        
    % Copyright 2018 The Mathworks, Inc.

    methods
        function this = ConverterForMaxUnpooling2DLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, parameterInputs, networkInputs, networkOutputs, layerMap] = toOnnx(this, TensorNameMap)
            import nnet.internal.cnn.onnx.*

            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            nodeProto           = NodeProto;
            nodeProto.op_type   = 'MaxUnpool';
            nodeProto.domain    = 'com.mathworks';
            nodeProto.name      = onnxName;
            nodeProto.input     = mapTensorNames(this, this.InputLayerNames(:)', TensorNameMap);
            nodeProto.output    = {onnxName};
            nodeProto.doc_string = 'maxUnpooling2dLayer';
            
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
