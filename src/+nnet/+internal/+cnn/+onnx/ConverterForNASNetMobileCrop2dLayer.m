classdef ConverterForNASNetMobileCrop2dLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a nnet.nasnetmobile.layer.NASNetMobileCrop2dLayer into ONNX
        
    % Copyright 2019 The Mathworks, Inc.
    
    methods
        function this = ConverterForNASNetMobileCrop2dLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, parameterInputs, networkInputs, networkOutputs, layerMap] = toOnnx(this, TensorNameMap)
            import nnet.internal.cnn.onnx.*
            
            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            nodeProto = NodeProto;
            nodeProto.op_type   = 'Slice';
            nodeProto.name      = onnxName;
            nodeProto.input     = mapTensorNames(this, this.InputLayerNames(1), TensorNameMap);
            nodeProto.output    = {onnxName};
            
            % Find Left, Top, Right, Bottom of resulting region
            XH                  = this.InputLayerSizes{1}(1);
            XW                  = this.InputLayerSizes{1}(2);
            ReferenceLayerSize  = this.InputLayerSizes{2};
            H                   = ReferenceLayerSize(1);
            W                   = ReferenceLayerSize(2);
            assert(isnumeric(this.NNTLayer.Location) && numel(this.NNTLayer.Location)==2); % [L T]
            T = this.NNTLayer.Location(2);
            L = this.NNTLayer.Location(1);
            B = T + H - 1;
            R = L + W - 1;
            
            % Convert from retained region to borders (number of rows and cols to delete):
            LB = L - 1;
            TB = T - 1;
            RB = XW - R;
            BB = XH - B;
            
            nodeProto.attribute = [...
                makeAttributeProto('starts', 'INTS', [0 0 TB LB]), ...
                makeAttributeProto('ends',   'INTS', [intmax intmax -BB -RB])];
            
            parameterInitializers   = [];
            parameterInputs         = [];
            networkInputs           = [];
            networkOutputs          = [];
            layerMap                = containers.Map;
            
            if nameChanged
                layerMap(this.NNTLayer.Name) = onnxName;
            end
        end
    end
end