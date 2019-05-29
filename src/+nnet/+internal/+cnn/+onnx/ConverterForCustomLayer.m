classdef ConverterForCustomLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert an unknown custom layer into ONNX
    
    % Copyright 2018 The Mathworks, Inc.
    
    properties
        LayerAnalyzer
    end
    
    methods
        function this = ConverterForCustomLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
            this.LayerAnalyzer = layerAnalyzer;
        end
        
        function [nodeProto, parameterInitializers, parameterInputs, networkInputs, networkOutputs, layerMap] = toOnnx(this, TensorNameMap)
            % An NNT unknown custom layer translates into a com.mathworks.Custom operator.
            import nnet.internal.cnn.onnx.*
            
            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            
            % Input names are the input tensors and learnable parameters of
            % the layer
            LearnableNames          = this.LayerAnalyzer.Learnables.Properties.RowNames(:)';
            LearnableONNXNames      = cellfun(@(name)legalizeNNTName(this, name), LearnableNames, 'UniformOutput', false);
            AllInputNames           = mapTensorNames(this, [this.InputLayerNames(:)', LearnableONNXNames], TensorNameMap);
            
            % Output names are the output tensors of the layer. If the only
            % output is 'out', change it to the layer name
            OutputNames             = this.LayerAnalyzer.Outputs.Properties.RowNames(:)';
            if isequal(OutputNames, {'out'})
                OutputNames         = {onnxName};
            end
            OutputONNXNames         = cellfun(@(name)legalizeNNTName(this, name), OutputNames, 'UniformOutput', false);
            
            % Make the attributes
            HyperNames              = this.LayerAnalyzer.Hypers.Properties.RowNames(:)';
            HyperONNXNames          = cellfun(@(name)legalizeNNTName(this, name), HyperNames, 'UniformOutput', false);
            HyperValues             = this.LayerAnalyzer.Hypers.Value(:)';
            Attributes              = cellfun(@attributeFromHyper, HyperONNXNames, HyperValues);
            if isempty(Attributes)
                Attributes = [];
            end
            
            % Make the nodeProto
            nodeProto               = NodeProto;
            nodeProto.op_type       = 'Custom';
            nodeProto.domain        = 'com.mathworks';
            nodeProto.name          = onnxName;
            nodeProto.input         = AllInputNames;
            nodeProto.output        = OutputONNXNames;
            nodeProto.doc_string	= getString(message('nnet_cnn_onnx:onnx:CustomLayerDocString'));
            nodeProto.attribute     = Attributes;
            
            % Make Initializers for learnable parameters
            parameterInitializers   = cellfun(@(OrigParamName, ONNXParamName)makeInitializer(this.NNTLayer, OrigParamName, ONNXParamName),...
                LearnableNames, LearnableONNXNames);
            
            % Make parameter Inputs
            parameterInputs         = arrayfun(@makeValueInfoProtoFromTensorProto, parameterInitializers);
            
            networkInputs           = [];
            networkOutputs          = [];
            layerMap                = containers.Map;
            
            if nameChanged
                layerMap(this.NNTLayer.Name) = onnxName;
            end
            % Add output names to layerMap
            for i=1:numel(OutputNames)
                layerMap(OutputNames{i}) = OutputONNXNames{i};
            end
        end
    end
end

function initializer = makeInitializer(NNTLayer, OrigParamName, ONNXParamName)
import nnet.internal.cnn.onnx.*
ParamValue                  = NNTLayer.(OrigParamName);
initializer                 = TensorProto;
initializer.name            = ONNXParamName;
initializer.dims            = dimVector(size(ParamValue), numel(size(ParamValue)));
if isnumeric(ParamValue)
    initializer.data_type	= TensorProto_DataType.FLOAT;
    initializer.raw_data    = rawData(single(ParamValue));
else
    initializer.data_type	= TensorProto_DataType.STRING;
    initializer.raw_data    = rawData(char(ParamValue));
end
end

function attribute = attributeFromHyper(HyperONNXName, HyperValue)
import nnet.internal.cnn.onnx.*
if isnumeric(HyperValue)
    attribute = makeAttributeProto(HyperONNXName, 'FLOATS', HyperValue);
else
    attribute = makeAttributeProto(HyperONNXName, 'STRING', char(HyperValue));
end
end