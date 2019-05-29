classdef ConverterForAffineLayersBaseClass < nnet.internal.cnn.onnx.NNTLayerConverter
    % A DLT layer that does Y=aX+b with scalar (a,b). If b~=0, it is
    % exported as Mul(a) followed by Add(b). If b==0. it is just Mul(a).
    
    % Copyright 2018 The Mathworks, Inc.
    
    
    properties
        a
        b
    end
    
    methods
        function this = ConverterForAffineLayersBaseClass(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function this = set.a(this, a)
            assert(isnumeric(a) && isscalar(a));
            this.a = a;
        end
        
        function this = set.b(this, b)
            assert(isnumeric(b) && isscalar(b));
            this.b = b;
        end
        
        function [nodeProto, parameterInitializers, parameterInputs, networkInputs, networkOutputs, layerMap] = toOnnx(this, TensorNameMap)
            import nnet.internal.cnn.onnx.*
            % Implementation of Y=aX+b: Mul(a) followed by Add(b).
            
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
            % Initializer for B
            t1              = TensorProto;
            t1.name         = MulBName;
            t1.data_type    = TensorProto_DataType.FLOAT;
            t1.raw_data     = rawData(single(this.a));
            t1.dims         = dimVector(1,1);
            parameterInitializers(1) = t1;
            parameterInputs(1)       = arrayfun(@makeValueInfoProtoFromTensorProto, parameterInitializers(1));
            
            % (2) If b~=0, also generate an Add operator: Adds two inputs A
            % and B. We'll set A to be the output of the Mul and set B to
            % be the bias. We'll give B an initializer.
            if this.b ~= 0
                AddNodeName            = [onnxName, '_Add'];
                AddBName               = [AddNodeName, '_B'];
                nodeProto(2)           = NodeProto;
                nodeProto(2).op_type   = 'Add';
                nodeProto(2).name      = AddNodeName;
                nodeProto(2).input     = {MulNodeName, AddBName};
                nodeProto(2).output    = {AddNodeName};
                if this.OpsetVersion < 7
                    nodeProto(2).attribute = makeAttributeProto('broadcast',	'INT', 1);
                end
                % Initializer for B
                t2              = TensorProto;
                t2.name         = AddBName;
                t2.data_type    = TensorProto_DataType.FLOAT;
                t2.raw_data     = rawData(single(this.b));
                t2.dims      	= dimVector(1,1);
                
                parameterInitializers(2) = t2;
                parameterInputs(2)       = arrayfun(@makeValueInfoProtoFromTensorProto, parameterInitializers(2));
            end
            
            networkInputs        	= [];
            networkOutputs        	= [];
            layerMap                = containers.Map;
            
            if this.b ~= 0
                % If there was a bias, set the output tensor to be the Add node
                layerMap(this.NNTLayer.Name) = AddNodeName;
            else
                % If there was no bias, set the output tensor to be the Mul node
                layerMap(this.NNTLayer.Name) = MulNodeName;
            end
        end
    end
end
