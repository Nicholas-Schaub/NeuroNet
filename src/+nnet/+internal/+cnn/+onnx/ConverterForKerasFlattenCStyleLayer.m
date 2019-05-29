classdef ConverterForKerasFlattenCStyleLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a nnet.keras.layer.FlattenCStyleLayer into ONNX
    
    % Copyright 2018 The Mathworks, Inc.
    
    methods
        function this = ConverterForKerasFlattenCStyleLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, parameterInputs, networkInputs, networkOutputs, layerMap] = toOnnx(this, TensorNameMap)
            % Generate a Transpose followed by a Flatten. The ONNX input
            % tensor is [N C H W]. To match the order of the units in  the
            % flattened MATLAB layer we need [H W C] within each N, or [N H
            % W C]. After that, we flatten with axis=1 to give [N HWC] in
            % row-major, as desired.
            import nnet.internal.cnn.onnx.*
            
            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            % (1) Transpose (permute)
            TransposeNodeName       = [onnxName '_Transpose'];
            nodeProto(1)            = NodeProto;
            nodeProto(1).op_type    = 'Transpose';
            nodeProto(1).name     	= TransposeNodeName;
            nodeProto(1).input      = mapTensorNames(this, this.InputLayerNames(1), TensorNameMap);
            nodeProto(1).output     = {TransposeNodeName};
            perm                    = [0 2 3 1];        % Permute [N C H W] into [N H W C]
            nodeProto(1).attribute  = makeAttributeProto('perm', 'INTS', perm);
            
            % (2) Flatten
            FlattenNodeName         = [onnxName '_Flatten'];
            nodeProto(2)            = NodeProto;
            nodeProto(2).op_type    = 'Flatten';
            nodeProto(2).name       = FlattenNodeName;
            nodeProto(2).input      = {TransposeNodeName};
            nodeProto(2).output     = {FlattenNodeName};
            axis                    = 1;
            nodeProto(2).attribute  = makeAttributeProto('axis', 'INT', axis);
            
            parameterInitializers   = [];
            parameterInputs         = [];
            networkInputs           = [];
            networkOutputs          = [];
            layerMap                = containers.Map;
            
            % Set the output tensor name to the Flatten name
            layerMap(this.NNTLayer.Name) = FlattenNodeName;
        end
    end
end

