classdef StringStringEntryProto
    %   optional string key = 1;
    %   optional string value= 2;
    properties
        key
        value
    end
    
    methods
        function this = StringStringEntryProto(varargin)
            import nnet.internal.cnn.onnx.*
            if nargin > 0
                Ptr = varargin{1};
                % Get raw properties
                PropertyCell = onnxmex(int32(FuncName.EdecodeStringStringEntryProto), Ptr);
                [this.key, this.value] = PropertyCell{:};
                % Call constructors on properties that are Proto objects
                % (none)
            end
        end
        
        function encodeStringStringEntryProto(this, CPtr)
            % Recursively fill the CPtr from 'this'.
            import nnet.internal.cnn.onnx.*
            PropertyCell = {this.key, this.value};
            PtrCell = onnxmex(int32(FuncName.EencodeStringStringEntryProto), CPtr, PropertyCell);
        end
    end
end
