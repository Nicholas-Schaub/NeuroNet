classdef ConverterForImageStandardizationLayersBaseClass < nnet.internal.cnn.onnx.NNTLayerConverter
    % A DLT layer that does Y = Scale.*X + Offset where X is [H W C N], and
    % Scale and Offset are Cx1. If Offset~=0, it is exported as Mul(Scale)
    % followed by Add(Offset). If Offset==0. it is just Mul(Scale).
    
    % Copyright 2018 The Mathworks, Inc.
    
    properties
        Scale
        Offset
    end
    
    methods
        function this = ConverterForImageStandardizationLayersBaseClass(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function this = set.Scale(this, Scale)
            assert(isnumeric(Scale) && iscolumn(Scale));
            this.Scale = Scale;
        end
        
        function this = set.Offset(this, Offset)
            assert(isnumeric(Offset) && iscolumn(Offset));
            this.Offset = Offset;
        end
        
        function [nodeProto, parameterInitializers, parameterInputs, networkInputs, networkOutputs, layerMap] = toOnnx(this, TensorNameMap)
            import nnet.internal.cnn.onnx.*
            % Implementation of Y=Scale.*X+Offset: Mul(Scale) followed by Add(Offset).
            
            C = numel(this.Scale);
            
            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            % (1) Mul
            MulNodeName            = [onnxName, '_Mul'];
            MulBName               = [MulNodeName '_B'];
            nodeProto(1)           = NodeProto;
            nodeProto(1).op_type   = 'Mul';
            nodeProto(1).name      = MulNodeName;
            nodeProto(1).input     = mapTensorNames(this, {this.InputLayerNames{1}, MulBName}, TensorNameMap);
            nodeProto(1).output    = {MulNodeName};
            if this.OpsetVersion < 7
                nodeProto(1).attribute = makeAttributeProto('broadcast', 'INT', 1);
            end
            % Initializer for Scale
            t1              = TensorProto;
            t1.name         = MulBName;
            t1.data_type    = TensorProto_DataType.FLOAT;
            t1.raw_data     = rawData(single(this.Scale));
            t1.dims         = dimVector([1 C 1 1], 4);      % ONNX images have shape [N C H W]
            
            parameterInitializers(1) = t1;
            parameterInputs(1)       = arrayfun(@makeValueInfoProtoFromTensorProto, parameterInitializers(1));
            
            % (2) If Offset~=0, also generate an Add operator: Adds two inputs Scale
            % and Offset. We'll set Scale to be the output of the Mul and set Offset to
            % be the bias. We'll give Offset an initializer.
            if any(this.Offset ~= 0)
                AddNodeName            = [onnxName, '_Add'];
                AddBName               = [AddNodeName, '_B'];
                nodeProto(2)           = NodeProto;
                nodeProto(2).op_type   = 'Add';
                nodeProto(2).name      = AddNodeName;
                nodeProto(2).input     = {MulNodeName, AddBName};
                nodeProto(2).output    = {AddNodeName};
                if this.OpsetVersion < 7
                    nodeProto(2).attribute = makeAttributeProto('broadcast', 'INT', 1);
                end
                % Initializer for Offset
                t2              = TensorProto;
                t2.name         = AddBName;
                t2.data_type    = TensorProto_DataType.FLOAT;
                t2.raw_data     = rawData(single(this.Offset));
                t2.dims      	= dimVector([1 C 1 1], 4);      % ONNX images have shape [N C H W]
                
                parameterInitializers(2) = t2;
                parameterInputs(2)       = arrayfun(@makeValueInfoProtoFromTensorProto, parameterInitializers(2));
            end
            
            networkInputs        	= [];
            networkOutputs        	= [];
            layerMap                = containers.Map;
            
            if any(this.Offset ~= 0)
                % If there was an Offset, set the output tensor to be the Add node
                layerMap(this.NNTLayer.Name) = AddNodeName;
            else
                % If there was no Offset, set the output tensor to be the Mul node
                layerMap(this.NNTLayer.Name) = MulNodeName;
            end
        end
    end
end
