classdef ConverterForSoftmaxLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a SoftmaxLayer into ONNX
    
    % Copyright 2018 The Mathworks, Inc.
    
    methods
        function this = ConverterForSoftmaxLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, parameterInputs, networkInputs, networkOutputs, layerMap] = toOnnx(this, TensorNameMap)
            import nnet.internal.cnn.onnx.*
            
            parameterInitializers   = [];
            parameterInputs         = [];
            networkInputs           = [];
            networkOutputs          = [];
            layerMap                = containers.Map;
            
            
            [ONNXName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            OutputSize = this.OutputSize{1};
            if iIsFlattened(OutputSize)
                % Generate a single Softmax node
                nodeProto           = NodeProto;
                nodeProto.op_type	= 'Softmax';
                nodeProto.name      = ONNXName;
                nodeProto.input     = mapTensorNames(this, this.InputLayerNames(1), TensorNameMap);
                nodeProto.output    = {ONNXName};
                layerMap(this.NNTLayer.Name) = ONNXName;
            else
                assert(numel(OutputSize)==3); % Shape is [H W C]
                % Generate a sequence of [Transpose, Softmax, Transpose] nodes
                % In ONNX, Transpose NCWH to NWHC:
                Transpose1NodeName     = [ONNXName '_Transpose1'];
                nodeProto(1)           = NodeProto;
                nodeProto(1).op_type   = 'Transpose';
                nodeProto(1).name      = Transpose1NodeName;
                nodeProto(1).input     = mapTensorNames(this, this.InputLayerNames(1), TensorNameMap);
                nodeProto(1).output    = {Transpose1NodeName};
                nodeProto(1).attribute = [...
                    makeAttributeProto('perm', 'INTS', [0 2 3 1]),...
                    ];
                % Softmax the last dimension of NWHC
                SoftmaxName            = [ONNXName '_Softmax'];
                nodeProto(2)           = NodeProto;
                nodeProto(2).op_type   = 'Softmax';
                nodeProto(2).name      = SoftmaxName;
                nodeProto(2).input     = {Transpose1NodeName};
                nodeProto(2).output    = {SoftmaxName};
                nodeProto(2).attribute = [...
                    makeAttributeProto('axis', 'INT', 3),...    % 'axis=3' means that 3 dimensions go on the lefthand side of a temporary reshape: [NWH,C]
                    ];
                % Transpose NWHC back to NCWH:
                Transpose2NodeName     = [ONNXName '_Transpose2'];
                nodeProto(3)           = NodeProto;
                nodeProto(3).op_type   = 'Transpose';
                nodeProto(3).name      = Transpose2NodeName;
                nodeProto(3).input     = {SoftmaxName};
                nodeProto(3).output    = {Transpose2NodeName};
                nodeProto(3).attribute = [...
                    makeAttributeProto('perm', 'INTS', [0 3 1 2]),...
                    ];
                % Set the output tensor name to the last node's output name
                layerMap(this.NNTLayer.Name) = Transpose2NodeName;
            end
        end
    end
end

function tf = iIsFlattened(OutputSize)
% Return true if its shape is a scalar or [1 1 C]
tf = isscalar(OutputSize) || (numel(OutputSize) == 3 && isequal(OutputSize(1:2), [1 1]));
end
