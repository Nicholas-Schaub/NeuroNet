classdef ConverterForONNXFlattenLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a nnet.onnx.layer.FlattenLayer into ONNX
    
    % Copyright 2018 The Mathworks, Inc.
    
    methods
        function this = ConverterForONNXFlattenLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, parameterInputs, networkInputs, networkOutputs, layerMap] = toOnnx(this, TensorNameMap)
            % Since this layer is just the MATLAB version of ONNX's
            % Flatten, we can just generate a Flatten.
            import nnet.internal.cnn.onnx.*
            
            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            % (1) Transpose (permute)
            NodeName                = onnxName;
            nodeProto(1)            = NodeProto;
            nodeProto(1).op_type    = 'Flatten';
            nodeProto(1).name     	= NodeName;
            nodeProto(1).input      = mapTensorNames(this, this.InputLayerNames(1), TensorNameMap);
            nodeProto(1).output     = {NodeName};
            axis                    = 1;
            nodeProto(1).attribute  = makeAttributeProto('axis', 'INT', axis);
            
            parameterInitializers   = [];
            parameterInputs         = [];
            networkInputs           = [];
            networkOutputs          = [];
            layerMap                = containers.Map;
            
            % Maybe update the layer name map
            if nameChanged
                layerMap(this.NNTLayer.Name) = NodeName;
            end
        end
    end
end

