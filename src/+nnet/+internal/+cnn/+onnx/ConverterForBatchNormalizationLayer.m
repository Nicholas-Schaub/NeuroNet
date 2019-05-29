classdef ConverterForBatchNormalizationLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a batchNormalizationLayer into ONNX
    
    % Copyright 2018 The Mathworks, Inc.
    
    methods
        function this = ConverterForBatchNormalizationLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, parameterInputs, networkInputs, networkOutputs, layerMap] = toOnnx(this, TensorNameMap)
            import nnet.internal.cnn.onnx.*
            
            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            % Make the nodeProto
            nodeProto           = NodeProto;
            nodeProto.op_type	= 'BatchNormalization';
            nodeProto.name      = onnxName;
            nodeProto.input     = mapTensorNames(this,{this.InputLayerNames{1}, [onnxName '_scale'], [onnxName '_B'], [onnxName '_mean'], [onnxName '_var']}, TensorNameMap);
            nodeProto.output    = {onnxName};
            nodeProto.attribute = makeAttributeProto('epsilon', 'FLOAT', this.NNTLayer.Epsilon);
            if this.OpsetVersion < 7
                nodeProto.attribute(1,2) = makeAttributeProto('is_test', 'INT', 1);
            end
            
            % Make parameter Initializers for: scale, B, mean, var
            t1 = TensorProto;
            t1.name = [onnxName '_scale'];
            t1.data_type = TensorProto_DataType.FLOAT;
            t1.raw_data = rawData(single(this.NNTLayer.Scale));
            t1.dims = dimVector(numel(this.NNTLayer.Scale),1);           % NNT data: 1-1-numChannels
            
            t2 = TensorProto;
            t2.name = [onnxName '_B'];
            t2.data_type = TensorProto_DataType.FLOAT;
            t2.raw_data = rawData(single(this.NNTLayer.Offset));
            t2.dims = dimVector(numel(this.NNTLayer.Offset),1);          % NNT data: 1-1-numChannels
            
            t3 = TensorProto;
            t3.name = [onnxName '_mean'];
            t3.data_type = TensorProto_DataType.FLOAT;
            t3.raw_data = rawData(single(this.NNTLayer.TrainedMean));
            t3.dims = dimVector(numel(this.NNTLayer.TrainedMean),1);  	% NNT data: 1-1-numChannels
            
            t4 = TensorProto;
            t4.name = [onnxName '_var'];
            t4.data_type = TensorProto_DataType.FLOAT;
            t4.raw_data = rawData(single(this.NNTLayer.TrainedVariance));
            t4.dims = dimVector(numel(this.NNTLayer.TrainedVariance),1);	% NNT data: 1-1-numChannels
            
            parameterInitializers = [t1 t2 t3 t4];
            
            % Make parameter Inputs
            parameterInputs = arrayfun(@makeValueInfoProtoFromTensorProto, parameterInitializers);
            
            networkInputs           = [];
            networkOutputs          = [];
            layerMap                = containers.Map;
            
            if nameChanged
                layerMap(this.NNTLayer.Name) = onnxName;
            end
        end
    end
end

