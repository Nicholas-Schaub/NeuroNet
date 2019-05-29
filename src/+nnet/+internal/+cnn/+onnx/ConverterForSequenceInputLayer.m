classdef ConverterForSequenceInputLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a SequenceInputLayer into ONNX
        
    % Copyright 2018 The Mathworks, Inc.

    methods
        function this = ConverterForSequenceInputLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, parameterInputs, networkInputs, networkOutputs, layerMap] = toOnnx(this, TensorNameMap)
            import nnet.internal.cnn.onnx.*

            inputSequenceTensorName = legalizeNNTName(this, this.NNTLayer.Name);
            nodeProto               = [];
            parameterInitializers   = [];
            parameterInputs         = [];
            % NNT size is a scalar inputDim, ONNX LSTM input 'X' shape is
            % [seqLen, batchSize, inputDim]
            % (https://github.com/onnx/onnx/blob/master/docs/Operators.md#LSTM)
            networkInputs           = makeValueInfoProtoFromDimensions(...
                inputSequenceTensorName, ...
                TensorProto_DataType.FLOAT, ...
                [10 1 this.NNTLayer.InputSize]);  
            networkOutputs          = [];
            layerMap                = containers.Map;
        end
    end
end
