classdef TypeProto_Tensor
    %     optional TensorProto.DataType elem_type = 1;
    %     optional TensorShapeProto shape = 2;
    properties
        elem_type
        shape
    end
    
    methods
        function this = TypeProto_Tensor(varargin)
            import nnet.internal.cnn.onnx.*
            if nargin > 0
                Ptr = varargin{1};
                % Get raw properties
                PropertyCell = onnxmex(int32(FuncName.EdecodeTypeProto_Tensor), Ptr);
                [this.elem_type, this.shape] = PropertyCell{:};
                % Call constructors on properties that are Proto objects
                if ~isempty(this.elem_type)
                    this.elem_type = TensorProto_DataType(this.elem_type);
                end
                if ~isempty(this.shape)
                    this.shape = TensorShapeProto(this.shape);
                end
            end
        end
        
        function encodeTypeProto_Tensor(this, CPtr)
            % Recursively fill the CPtr from 'this'.
            import nnet.internal.cnn.onnx.*
            PropertyCell = {int32(this.elem_type), this.shape};
            PtrCell = onnxmex(int32(FuncName.EencodeTypeProto_Tensor), CPtr, PropertyCell);
            % Fill pointer objects
            arrayfun(@encodeTensorShapeProto, this.shape, PtrCell{2});
        end
    end
end