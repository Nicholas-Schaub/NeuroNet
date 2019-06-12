classdef TypeProto_Sequence
    %     optional TypeProto elem_type = 1;
    properties
        elem_type
    end
    
    methods
        function this = TypeProto_Sequence(varargin)
            import nnet.internal.cnn.onnx.*
            if nargin > 0
                Ptr = varargin{1};
                % Get raw properties
                PropertyCell = onnxmex(int32(FuncName.EdecodeTypeProto_Sequence), Ptr);
                [this.elem_type] = PropertyCell{:};
                % Call constructors on properties that are Proto objects
                if ~isempty(this.elem_type)
                    this.elem_type = TypeProto(this.elem_type);
                end
            end
        end
        
        function encodeTypeProto_Sequence(this, CPtr)
            % Recursively fill the CPtr from 'this'.
            import nnet.internal.cnn.onnx.*
            PropertyCell = {this.elem_type};
            PtrCell = onnxmex(int32(FuncName.EencodeTypeProto_Sequence), CPtr, PropertyCell);
            % Fill pointer objects
            arrayfun(@encodeTypeProto, this.elem_type, PtrCell{1});
        end
    end
end