classdef ConverterForKerasZeroPadding2dLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a nnet.keras.layer.ZeroPadding2dLayer into ONNX
        
    % Copyright 2018 The Mathworks, Inc.

    methods
        function this = ConverterForKerasZeroPadding2dLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, parameterInputs, networkInputs, networkOutputs, layerMap] = toOnnx(this, TensorNameMap)
            import nnet.internal.cnn.onnx.*

            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            nodeProto = NodeProto;
            nodeProto.op_type   = 'Pad';
            nodeProto.name      = onnxName;
            nodeProto.input     = mapTensorNames(this, this.InputLayerNames(1), TensorNameMap);
            nodeProto.output    = {onnxName};
            
            % Assuming the input tensor is [N C H W]
            pads = [0, 0, this.NNTLayer.Top, this.NNTLayer.Left, 0, 0, this.NNTLayer.Bottom, this.NNTLayer.Right];
            
            nodeProto.attribute = [...
                makeAttributeProto('mode', 'STRING', 'constant'),...
                makeAttributeProto('pads', 'INTS', pads),...
                makeAttributeProto('value', 'FLOAT', 0)
                ];
            
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
