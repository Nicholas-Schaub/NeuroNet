classdef ConverterForDepthConcatenationLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a DepthConcatenationLayer into ONNX
        
    % Copyright 2018 The Mathworks, Inc.
    
    methods
        function this = ConverterForDepthConcatenationLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, parameterInputs, networkInputs, networkOutputs, layerMap] = toOnnx(this, TensorNameMap)
            % nodeProto is a NodeProto.
            % parameterInitializers is a TensorProto array.           
            % parameterInputs is a ValueInfoProto array describing the parameters of the node.
            import nnet.internal.cnn.onnx.*

            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            nodeProto = NodeProto;
            nodeProto.op_type   = 'Concat';
            nodeProto.name      = onnxName;
            nodeProto.input     = mapTensorNames(this, this.InputLayerNames(:)', TensorNameMap);
            nodeProto.output    = {onnxName};
            
            concatAxis          = 1;
            nodeProto.attribute = [...
                makeAttributeProto('axis', 'INT', concatAxis)
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
