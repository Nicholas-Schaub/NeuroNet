classdef ConverterForROIMaxPooling2DLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a roiMaxPooling2DLayer into ONNX
    
    % Copyright 2018 The Mathworks, Inc.
    
    methods
        function this = ConverterForROIMaxPooling2DLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, parameterInputs, networkInputs, networkOutputs, layerMap] = toOnnx(this, TensorNameMap)
            import nnet.internal.cnn.onnx.*
            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            OutputSize      = this.NNTLayer.OutputSize;
            internalLayer   = iGetInternalLayer(this.NNTLayer);
            ScaleFactor     = internalLayer.ScaleFactor;
            if ScaleFactor(1) ~= ScaleFactor(2)
                warning(message('nnet_cnn_onnx:onnx:ROIMaxPooling2DLayerScalesUnequal', ...
                    this.NNTLayer.Name, ScaleFactor(1), ScaleFactor(2), ScaleFactor(1)));
            end
            ScaleFactor = ScaleFactor(1);

            nodeProto           = NodeProto;
            nodeProto.op_type   = 'MaxRoiPool';
            nodeProto.name      = onnxName;
            nodeProto.input     = mapTensorNames(this, this.InputLayerNames, TensorNameMap);
            nodeProto.output  	= {onnxName};
            layerMap        	= containers.Map;
            if nameChanged
                layerMap(this.NNTLayer.Name) = onnxName;
            end
            
            nodeProto.attribute = [...
                makeAttributeProto('pooled_shape',  'INTS', OutputSize),...
                makeAttributeProto('spatial_scale', 'FLOAT', ScaleFactor),...
                ];
            parameterInitializers   = [];
            parameterInputs         = [];
            networkInputs           = [];
            networkOutputs          = [];
        end
    end
end

function L = iGetInternalLayer(ExtLayer)
C = nnet.cnn.layer.Layer.getInternalLayers(ExtLayer);
L = C{1};
end