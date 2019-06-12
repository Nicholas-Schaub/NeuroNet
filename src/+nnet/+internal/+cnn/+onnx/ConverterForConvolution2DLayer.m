classdef ConverterForConvolution2DLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a convolution2dLayer into ONNX
        
    % Copyright 2018 The Mathworks, Inc.
    
    methods
        function this = ConverterForConvolution2DLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, parameterInputs, networkInputs, networkOutputs, layerMap] = toOnnx(this, TensorNameMap)
            import nnet.internal.cnn.onnx.*
            
            % Make the nodeProto
            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            WName               = [onnxName '_W'];
            BName               = [onnxName '_B'];
            nodeProto           = NodeProto;
            nodeProto.op_type	= 'Conv';
            nodeProto.name      = onnxName;
            nodeProto.input     = mapTensorNames(this, {this.InputLayerNames{1}, WName, BName}, TensorNameMap);
            nodeProto.output    = {onnxName};
            nntWeightSize      	= size(this.NNTLayer.Weights);
            kernel_shape        = nntWeightSize(1:2);
            nodeProto.attribute = [...
                makeAttributeProto('group',        'INT',  numel(this.NNTLayer.NumChannels)),... % See alexnet for an example where this is >1
                makeAttributeProto('dilations',    'INTS', this.NNTLayer.DilationFactor),...
                makeAttributeProto('kernel_shape', 'INTS', kernel_shape),...
                makeAttributeProto('pads',         'INTS', this.NNTLayer.PaddingSize([1,3,2,4])),...
                makeAttributeProto('strides',      'INTS', this.NNTLayer.Stride)
                ];
            
            % Make parameter Initializers for: W, B
            t1              = TensorProto;
            t1.name         = WName;     
            t1.data_type	= TensorProto_DataType.FLOAT;
            permutedW       = permute(this.NNTLayer.Weights, [4 3 1 2]);	% NNT is HWCF. ONNX is FCHW.
            t1.raw_data     = rawData(single(permutedW));
            t1.dims         = dimVector(size(permutedW),4);           
            
            t2              = TensorProto;
            t2.name         = BName;
            t2.data_type    = TensorProto_DataType.FLOAT;
            t2.raw_data     = rawData(single(squeeze(this.NNTLayer.Bias)));
            t2.dims         = dimVector(numel(this.NNTLayer.Bias),1);            % NNT data: 1-1-numFilters
            
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

