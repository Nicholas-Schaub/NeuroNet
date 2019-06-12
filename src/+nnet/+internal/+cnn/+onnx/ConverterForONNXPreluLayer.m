classdef ConverterForONNXPreluLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a nnet.onnx.layer.PreluLayer into ONNX
    
    % Copyright 2018-2019 The Mathworks, Inc.
    
    methods
        function this = ConverterForONNXPreluLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, parameterInputs, networkInputs, networkOutputs, layerMap] = toOnnx(this, TensorNameMap)
            import nnet.internal.cnn.onnx.*
            
            [NodeName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            SlopeName         	= [NodeName '_Slope'];
            nodeProto          	= NodeProto;
            nodeProto.op_type	= 'PRelu';
            nodeProto.name     	= NodeName;
            nodeProto.input   	= mapTensorNames(this, {this.InputLayerNames{1}, SlopeName}, TensorNameMap);
            nodeProto.output  	= {NodeName};
                                   
            % Make parameter Initializer for slope
            t1              = TensorProto;
            t1.name         = SlopeName;     
            t1.data_type	= TensorProto_DataType.FLOAT;
            slope           = this.NNTLayer.Alpha(:);
            t1.raw_data     = rawData(single(slope));
            % Set dimensions appropriate for Unidirectional broadcasting:
            if this.IsRecurrentNetwork
                t1.dims     = dimVector(numel(slope),1);    % Leave at length 1: [T N C] + [C] = [T N C]
            else
                t1.dims     = dimVector(numel(slope),3);    % Extend to length 3: [N C H W] + [C 1 1] = [N C H W]
            end
            
            parameterInitializers   = [t1];
            % Make parameter Inputs
            parameterInputs         = arrayfun(@makeValueInfoProtoFromTensorProto, parameterInitializers);
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

