classdef ConverterForUnsupportedLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert an unknown layer into an ONNX
    % com.mathworks.Placeholder operator.
    %
    % For example:
    % nodeProto = 
    %   NodeProto with properties:
    %          input: {'fc1'  'fc2'  'fc3'  'fc4'}
    %         output: {'add4to2_out1'  'add4to2_out2'}
    %           name: 'add4to2'
    %        op_type: 'Placeholder'
    %         domain: 'com.mathworks'
    %      attribute: []
    %     doc_string: 'Placeholder operator'
    
    % Copyright 2018 The Mathworks, Inc.
    
    properties
        LayerAnalyzer
    end
    
    methods
        function this = ConverterForUnsupportedLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
            this.LayerAnalyzer = layerAnalyzer;
        end
        
        function [nodeProto, parameterInitializers, parameterInputs, networkInputs, networkOutputs, layerMap] = toOnnx(this, TensorNameMap)
            % An NNT unknown layer translates into a com.mathworks.Placeholder operator.
            import nnet.internal.cnn.onnx.*
            
            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            AllInputNames           = mapTensorNames(this, this.InputLayerNames(:)', TensorNameMap);
            
            % Output names are the output tensors of the layer. Prepend the
            % layer name and '/'. If the only output is 'out', use the
            % layer name alone. Finally, legalize the names for ONNX.
            OutputNames             = this.LayerAnalyzer.Outputs.Properties.RowNames(:)';
            if isequal(OutputNames, {'out'})
                OutputNames         = {onnxName};
            else
                OutputNames       	= cellfun(@(outName)[onnxName '/' outName], OutputNames, 'UniformOutput', false);
            end
            OutputONNXNames         = cellfun(@(name)legalizeNNTName(this, name), OutputNames, 'UniformOutput', false);
                        
            % Make the nodeProto
            nodeProto               = NodeProto;
            nodeProto.op_type       = 'Placeholder';
            nodeProto.domain        = 'com.mathworks';
            nodeProto.name          = onnxName;
            nodeProto.input         = AllInputNames;
            nodeProto.output        = OutputONNXNames;
            nodeProto.doc_string	= getString(message('nnet_cnn_onnx:onnx:PlaceholderOperatorDocString'));
            
            parameterInitializers   = [];
            parameterInputs         = [];
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
