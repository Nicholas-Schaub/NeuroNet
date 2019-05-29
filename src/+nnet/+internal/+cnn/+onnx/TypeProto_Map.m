classdef TypeProto_Map
    %     optional TensorProto.DataType key_type = 1;
    %     optional TypeProto value_type = 2;
    properties
        key_type
        value_type
    end
    
    methods
        function this = TypeProto_Map(varargin)
            import nnet.internal.cnn.onnx.*
            if nargin > 0
                Ptr = varargin{1};
                % Get raw properties
                PropertyCell = onnxmex(int32(FuncName.EdecodeTypeProto_Map), Ptr);
                [this.key_type, this.value_type] = PropertyCell{:};
                % Call constructors on properties that are Proto objects
                if ~isempty(this.key_type)
                    this.key_type = TensorProto_DataType(this.key_type);
                end
                if ~isempty(this.value_type)
                    this.value_type = TypeProto(this.value_type);
                end
            end
        end
        
        function encodeTypeProto_Map(this, CPtr)
            % Recursively fill the CPtr from 'this'.
            import nnet.internal.cnn.onnx.*
            PropertyCell = {int32(this.key_type), this.value_type};
            PtrCell = onnxmex(int32(FuncName.EencodeTypeProto_Map), CPtr, PropertyCell);
            % Fill pointer objects
            arrayfun(@TypeProto, this.value_type, PtrCell{2});
        end
    end
end
