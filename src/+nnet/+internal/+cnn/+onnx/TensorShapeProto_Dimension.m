classdef TensorShapeProto_Dimension
    %     oneof value {
    %       int64 dim_value = 1;
    %       string dim_param = 2;   // namespace Shape
    %     };
    properties
        dim_value
        dim_param
    end
    
    methods
        function this = TensorShapeProto_Dimension(varargin)
            import nnet.internal.cnn.onnx.*
            if nargin > 0
                Ptr = varargin{1};
                % Get raw properties
                PropertyCell = onnxmex(int32(FuncName.EdecodeTensorShapeProto_Dimension), Ptr);
                [this.dim_value, this.dim_param] = PropertyCell{:};
                % Call constructors on properties that are Proto objects
                % (none)
            end
        end
        
        function encodeTensorShapeProto_Dimension(this, CPtr)
            % Recursively fill the CPtr from 'this'. To implement 'oneof',
            % any field that is nonempty is writte. So make sure exactly
            % one of them is.
            import nnet.internal.cnn.onnx.*
            PropertyCell = {this.dim_value, this.dim_param};
            PtrCell = onnxmex(int32(FuncName.EencodeTensorShapeProto_Dimension), CPtr, PropertyCell);
        end
    end
end