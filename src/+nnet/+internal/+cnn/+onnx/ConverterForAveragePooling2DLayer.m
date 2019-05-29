classdef ConverterForAveragePooling2DLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert an averagePooling2dLayer into ONNX
        
    % Copyright 2018 The Mathworks, Inc.
    
    methods
        function this = ConverterForAveragePooling2DLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, parameterInputs, networkInputs, networkOutputs, layerMap] = toOnnx(this, TensorNameMap)
            import nnet.internal.cnn.onnx.*

            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            nodeProto = NodeProto;
            nodeProto.op_type   = 'AveragePool';
            nodeProto.name      = onnxName;
            nodeProto.input     = mapTensorNames(this, this.InputLayerNames(1), TensorNameMap);
            nodeProto.output    = {onnxName};
            
            nodeProto.attribute = [...
                makeAttributeProto('kernel_shape', 'INTS', this.NNTLayer.PoolSize),...
                makeAttributeProto('pads',         'INTS', this.NNTLayer.PaddingSize([1 3 2 4])),...   % DLT=tblr, ONNX=tlbr
                makeAttributeProto('strides',      'INTS', this.NNTLayer.Stride)
                ];
            
            if this.OpsetVersion >= 7
                nodeProto.attribute(end+1) = makeAttributeProto('count_include_pad', 'INT', 1);
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