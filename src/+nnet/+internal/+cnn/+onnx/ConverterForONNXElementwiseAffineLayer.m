classdef ConverterForONNXElementwiseAffineLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    
    % Copyright 2018 The Mathworks, Inc.
    
    methods
        function this = ConverterForONNXElementwiseAffineLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, parameterInputs, networkInputs, networkOutputs, layerMap] = toOnnx(this, TensorNameMap)
            import nnet.internal.cnn.onnx.*
            % Implementation of Y=X.*A+B: C=Mul(X,A) followed by Add(C,B).
            % Two initialiers added to model.
            
            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            needMul         = ~all(this.NNTLayer.Scale(:) == 1);
            needAdd         = ~all(this.NNTLayer.Offset(:) == 0) || ~needMul;  % Make sure there's at least 1 op.
            layerInputName  = mapTensorNames(this, this.InputLayerNames(1), TensorNameMap);
            layerInputName  = layerInputName{1};
            nodeIdx         = 0;
            % (1) Mul
            if needMul
                nodeIdx = nodeIdx+1;
                MulNodeName            = [onnxName, '_Mul'];
                MulConstName           = [MulNodeName '_B'];
                nodeProto(nodeIdx)           = NodeProto;
                nodeProto(nodeIdx).op_type   = 'Mul';
                nodeProto(nodeIdx).name      = MulNodeName;
                nodeProto(nodeIdx).input     = {layerInputName, MulConstName};
                nodeProto(nodeIdx).output    = {MulNodeName};
                if this.OpsetVersion < 7
                    % Include the Scale in CHW format:
                    nodeProto(nodeIdx).attribute  = [...
                        makeAttributeProto('broadcast', 'INT', 1),...
                        makeAttributeProto('axis',      'INT', 1),...
                        ];
                end
                % Initializer
                t1              = TensorProto;
                t1.name         = MulConstName;
                t1.data_type    = TensorProto_DataType.FLOAT;
                Scale           = permute(this.NNTLayer.Scale, [4 3 1 2]);	% From HWCN cm to NCHW cm
                t1.raw_data     = rawData(single(Scale));                   % NCHW rm
                t1.dims         = dimVector(size(Scale),4);                 % dims = NCHW
                if this.OpsetVersion < 7
                    % Include the Scale in CHW format:
                    t1.dims         = t1.dims(2:end);                           % dims = CHW
                end
                parameterInitializers(nodeIdx) = t1;
                parameterInputs(nodeIdx)       = arrayfun(@makeValueInfoProtoFromTensorProto, parameterInitializers(nodeIdx));
            end
            % (2) Add
            if needAdd
                nodeIdx = nodeIdx+1;
                if needMul
                    AddInputName = MulNodeName;
                else
                    AddInputName = layerInputName;
                end
                AddNodeName            = [onnxName, '_Add'];
                AddConstName           = [AddNodeName, '_B'];
                nodeProto(nodeIdx)           = NodeProto;
                nodeProto(nodeIdx).op_type   = 'Add';
                nodeProto(nodeIdx).name      = AddNodeName;
                nodeProto(nodeIdx).input     = {AddInputName, AddConstName};
                nodeProto(nodeIdx).output    = {AddNodeName};
                if this.OpsetVersion < 7
                    % Include the Offset in CHW format:
                    nodeProto(nodeIdx).attribute  = [...
                        makeAttributeProto('broadcast', 'INT', 1),...
                        makeAttributeProto('axis',      'INT', 1),...
                        ];
                end
                % Initializer
                t2              = TensorProto;
                t2.name         = AddConstName;
                t2.data_type    = TensorProto_DataType.FLOAT;
                Offset            = permute(this.NNTLayer.Offset, [4 3 1 2]);   % From HWCN cm to NCHW cm
                t2.raw_data     = rawData(single(Offset));                    % NCHW rm
                t2.dims         = dimVector(size(Offset),4);                  % dims = NCHW
                if this.OpsetVersion < 7
                    % Include the Offset in CHW format:
                    t2.dims         = t2.dims(2:end);                         	% dims = CHW
                end
                parameterInitializers(nodeIdx) = t2;
                parameterInputs(nodeIdx)       = arrayfun(@makeValueInfoProtoFromTensorProto, parameterInitializers(nodeIdx));
            end
            networkInputs        	= [];
            networkOutputs        	= [];
            layerMap                = containers.Map;
            if needAdd
                % Set the output tensor to be the Add node
                layerMap(this.NNTLayer.Name) = AddNodeName;
            elseif needMul
                % Set the output tensor to be the Mul node
                layerMap(this.NNTLayer.Name) = MulNodeName;
            end
        end
    end
end
