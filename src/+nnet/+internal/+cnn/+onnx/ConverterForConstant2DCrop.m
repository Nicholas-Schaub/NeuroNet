classdef ConverterForConstant2DCrop < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a Crop2DLayer into ONNX
        
    % Copyright 2018-2019 The Mathworks, Inc.
    
    methods
        function this = ConverterForConstant2DCrop(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, parameterInputs, networkInputs, networkOutputs, layerMap] = toOnnx(this, TensorNameMap)
            import nnet.internal.cnn.onnx.*
            
            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            nodeProto = NodeProto;
            nodeProto.op_type   = 'Slice';
            nodeProto.name      = onnxName;
            nodeProto.input     = mapTensorNames(this, this.InputLayerNames(1), TensorNameMap);
            nodeProto.output    = {onnxName};
            
            % Find Left, Top, Right, Bottom of resulting region
            InputLayerSize      = this.InputLayerSizes{1};
            H                   = InputLayerSize(1);
            W                   = InputLayerSize(2);
            T = this.NNTLayer.crop_ref(2);
            L = this.NNTLayer.crop_ref(1);

            nodeProto.attribute = [...
                makeAttributeProto('starts', 'INTS', [0 0 T L]), ...
                makeAttributeProto('ends',   'INTS', [intmax intmax 1-T 1-L])];
            
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
