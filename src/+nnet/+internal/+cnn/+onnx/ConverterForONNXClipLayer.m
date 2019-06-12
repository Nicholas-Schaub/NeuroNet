classdef ConverterForONNXClipLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a nnet.onnx.layer.ClipLayer into ONNX
    
    % Copyright 2018 The Mathworks, Inc.
    
    methods
        function this = ConverterForONNXClipLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, parameterInputs, networkInputs, networkOutputs, layerMap] = toOnnx(this, TensorNameMap)
            % Since this layer was derived from ONNX's Clip, we can just
            % generate a Clip.
            import nnet.internal.cnn.onnx.*
            
            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            NodeName                = onnxName;
            nodeProto(1)            = NodeProto;
            nodeProto(1).op_type    = 'Clip';
            nodeProto(1).name     	= NodeName;
            nodeProto(1).input      = mapTensorNames(this, this.InputLayerNames(1), TensorNameMap);
            nodeProto(1).output     = {NodeName};
                                   
            nodeProto(1).attribute  = [];
            if this.NNTLayer.Min > -Inf
                nodeProto(1).attribute = [nodeProto(1).attribute, makeAttributeProto('min', 'FLOAT', this.NNTLayer.Min)];
            end
            if this.NNTLayer.Max < Inf
                nodeProto(1).attribute = [nodeProto(1).attribute, makeAttributeProto('max', 'FLOAT', this.NNTLayer.Max)];
            end
            
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

