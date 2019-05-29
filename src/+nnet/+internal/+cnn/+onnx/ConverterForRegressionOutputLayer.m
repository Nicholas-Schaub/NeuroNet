classdef ConverterForRegressionOutputLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a RegressionOutputLayer into ONNX
        
    % Copyright 2018 The Mathworks, Inc.
    
    properties
        LayerAnalyzer
    end

    methods
        function this = ConverterForRegressionOutputLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
            this.LayerAnalyzer = layerAnalyzer;
        end
        
        function [nodeProto, parameterInitializers, parameterInputs, networkInputs, networkOutputs, layerMap] = toOnnx(this, TensorNameMap)
            import nnet.internal.cnn.onnx.*

            inputLayerName          = mapTensorNames(this, this.InputLayerNames(1), TensorNameMap);
            nodeProto               = [];
            parameterInitializers   = [];
            parameterInputs         = [];
            networkInputs           = [];
            dltSize                 = this.LayerAnalyzer.Inputs.Size{1}(:)';
            if this.IsRecurrentNetwork
                assert(isscalar(dltSize));
                outputTensorSize = [1 1 dltSize];           % DLT size is C. ONNX size is tnc with t=n=1.
            else
                assert(numel(dltSize)==3);
                outputTensorSize = [1 dltSize([3 1 2])];	% DLT size is hwc, ONNX size is nchw with n=1.
            end
            networkOutputs  = makeValueInfoProtoFromDimensions(inputLayerName{1}, TensorProto_DataType.FLOAT, outputTensorSize);
            layerMap    	= containers.Map;
        end
    end
end
