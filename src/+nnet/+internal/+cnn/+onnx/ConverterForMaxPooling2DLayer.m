classdef ConverterForMaxPooling2DLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a MaxPooling2dLayer into ONNX
    
    % Copyright 2018 The Mathworks, Inc.
    
    methods
        function this = ConverterForMaxPooling2DLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, parameterInputs, networkInputs, networkOutputs, layerMap] = toOnnx(this, TensorNameMap)
            import nnet.internal.cnn.onnx.*
            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            nodeProto           = NodeProto;
            nodeProto.op_type   = 'MaxPool';
            nodeProto.name      = onnxName;
            nodeProto.input     = mapTensorNames(this, this.InputLayerNames(1), TensorNameMap);
            if this.OpsetVersion >= 8 && this.NNTLayer.HasUnpoolingOutputs
                outName                 = [onnxName '_out'];
                indicesName             = [onnxName '_indices'];
                % Add renamed output tensors to name Map
                origOutName             = [this.NNTLayer.Name '/out'];
                origIndicesName         = [this.NNTLayer.Name '/indices'];
                layerMap(origOutName)   = outName;
                layerMap(origIndicesName) = indicesName;
                % Set node outputs
                nodeProto.output        = {outName, indicesName};
            else
                % No unpooling outputs
                layerMap                = containers.Map;
                if nameChanged
                    layerMap(this.NNTLayer.Name) = onnxName;
                end
                nodeProto.output        = {onnxName};
            end
            nodeProto.attribute = [...
                makeAttributeProto('kernel_shape',  'INTS', this.NNTLayer.PoolSize),...
                makeAttributeProto('pads',          'INTS', this.NNTLayer.PaddingSize([1 3 2 4])),...   % NNT=tblr, ONNX=tlbr
                makeAttributeProto('strides',       'INTS', this.NNTLayer.Stride)
                ];
            parameterInitializers   = [];
            parameterInputs         = [];
            networkInputs           = [];
            networkOutputs          = [];
        end
    end
end
