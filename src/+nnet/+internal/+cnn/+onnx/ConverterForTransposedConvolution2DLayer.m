classdef ConverterForTransposedConvolution2DLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a convolution2dLayer into ONNX
        
    % Copyright 2018 The Mathworks, Inc.

    methods
        function this = ConverterForTransposedConvolution2DLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, parameterInputs, networkInputs, networkOutputs, layerMap] = toOnnx(this, TensorNameMap)
            import nnet.internal.cnn.onnx.*
            
            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            % Make the nodeProto
            nodeProto           = NodeProto;
            nodeProto.op_type	= 'ConvTranspose';
            nodeProto.name      = onnxName;
            input               = {...
                this.InputLayerNames{1},... % X
                [onnxName '_W'],...
                [onnxName '_B']
                };
            nodeProto.input     = mapTensorNames(this, input(:)', TensorNameMap);
            nodeProto.output    = {onnxName};
            nntWeightSize      	= size(this.NNTLayer.Weights);
            kernel_shape        = nntWeightSize(1:2);           % HW
            cropVert            = this.NNTLayer.Cropping(1);
            cropHoriz           = this.NNTLayer.Cropping(2);
            nodeProto.attribute = [...
                makeAttributeProto('group',        'INT',  numel(this.NNTLayer.NumChannels)) ,...
                makeAttributeProto('dilations',    'INTS', [1 1]),...
                makeAttributeProto('kernel_shape', 'INTS', kernel_shape),...
                makeAttributeProto('pads',         'INTS', [cropVert, cropHoriz, cropVert, cropHoriz]),...  
                makeAttributeProto('strides',      'INTS', this.NNTLayer.Stride)
                ];
            
            % Make parameter Initializers for: W, B
            t1 = TensorProto;
            t1.name = [onnxName '_W'];     
            t1.data_type = TensorProto_DataType.FLOAT;
            permutedW = permute(this.NNTLayer.Weights, [4 3 1 2]);	% NNT is HWFC. ONNX is CFHW.
            t1.raw_data = rawData(single(permutedW));
            t1.dims = dimVector(size(permutedW),4);           
            
            t2 = TensorProto;
            t2.name = [onnxName '_B'];
            t2.data_type = TensorProto_DataType.FLOAT;
            t2.raw_data = rawData(single(squeeze(this.NNTLayer.Bias)));
            t2.dims = dimVector(numel(this.NNTLayer.Bias),1);            % NNT data: 1-1-numFilters
            
            parameterInitializers = [t1 t2];
            
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

