classdef ConverterForImageInputLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert an ImageInputLayer into ONNX
        
    % Copyright 2018 The Mathworks, Inc.
    
    methods
        function this = ConverterForImageInputLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, parameterInputs, networkInputs, networkOutputs, layerMap] = toOnnx(this, TensorNameMap)
            import nnet.internal.cnn.onnx.*
            % Generate a ValueInfoProto describing the input image tensor,
            % and opionally generate a Sub node to perform zerocenter
            % normalization.
            inputImageTensorName    = legalizeNNTName(this, this.NNTLayer.Name);
            averageImage            = this.NNTLayer.AverageImage;
            if isempty(averageImage)
                nodeProto        	= [];
                parameterInitializers = [];
            else
                % Make a 'Sub' node to subtract the average image
                subNodeName         = [inputImageTensorName '_Sub'];
                avgImgName          = [inputImageTensorName '_AvgImg'];
                nodeProto           = NodeProto;
                nodeProto.op_type	= 'Sub';
                nodeProto.name      = subNodeName;
                nodeProto.input     = {inputImageTensorName, avgImgName};
                nodeProto.output    = {nodeProto.name};
                if this.OpsetVersion < 7
                    nodeProto.attribute = [...
                        makeAttributeProto('broadcast', 'INT', 1),...
                        makeAttributeProto('axis', 'INT', 1)];
                end
                
                % Make a parameter Initializer for AvgImg
                t1                  = TensorProto;
                t1.name             = avgImgName;
                t1.data_type        = TensorProto_DataType.FLOAT;
                permutedImg         = permute(averageImage, [4 3 1 2]);     % NNT is hwcn, ONNX is nchw with n=1
                t1.raw_data         = rawData(single(permutedImg));
                t1.dims             = dimVector(size(permutedImg),4);       % dims = NCHW
                if this.OpsetVersion < 7
                    t1.dims             = t1.dims(2:end);                    	% dims = CHW
                end
                parameterInitializers = t1;
            end
            parameterInputs         = arrayfun(@makeValueInfoProtoFromTensorProto, parameterInitializers);
            
            % Make a ValueInfoProto describing the input image tensor
            networkInputs           = makeValueInfoProtoFromDimensions(...
                inputImageTensorName, ...
                TensorProto_DataType.FLOAT, ...
                [1 this.NNTLayer.InputSize([3 1 2])]);  % NNT is hwc, ONNX is nchw with n=1
            networkOutputs          = [];
            layerMap                = containers.Map;
            
            % If we created a Sub node, make all other layers refer to it
            % in their inputs instead of the input tensor.
            if ~isempty(averageImage)
                layerMap(this.NNTLayer.Name) = subNodeName;
            end
        end
    end
end
