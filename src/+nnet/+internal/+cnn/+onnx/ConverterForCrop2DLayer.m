classdef ConverterForCrop2DLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a Crop2DLayer into ONNX
        
    % Copyright 2018-2019 The Mathworks, Inc.
    
    methods
        function this = ConverterForCrop2DLayer(layerAnalyzer)
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
            InputLayerSize      = this.InputLayerSizes{1};
            XH                  = InputLayerSize(1);
            XW                  = InputLayerSize(2);
            ReferenceLayerSize  = this.InputLayerSizes{2};
            H                   = ReferenceLayerSize(1);
            W                   = ReferenceLayerSize(2);
            switch this.NNTLayer.Mode
                case 'centercrop'
                    assert(isequal(this.NNTLayer.Location, 'auto'));
                    sz           = InputLayerSize;
                    outputSize	 = ReferenceLayerSize(1:2);
                    % Compare the following to nnet.internal.cnn.layer.util.Crop2DCenterCropStrategy
                    centerX      = floor(sz(1:2)/2 + 1);
                    centerWindow = floor(outputSize/2 + 1);
                    offset       = centerX - centerWindow + 1;
                    T            = offset(1);
                    L            = offset(2);
                case 'custom'
                    assert(isnumeric(this.NNTLayer.Location) && numel(this.NNTLayer.Location)==2); % [L T]
                    T = this.NNTLayer.Location(2);
                    L = this.NNTLayer.Location(1);
                otherwise
                    assert(false);
            end
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
