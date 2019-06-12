classdef ValueInfoProto
    %   optional string name = 1;     // namespace Value
    %   optional TypeProto type = 2;
    %   optional string doc_string = 3;
    properties
        name
        type
        doc_string
    end
    
    methods
        function this = ValueInfoProto(varargin)
            import nnet.internal.cnn.onnx.*
            if nargin > 0
                Ptr = varargin{1};
                % Get raw properties
                PropertyCell = onnxmex(int32(FuncName.EdecodeValueInfoProto), Ptr);
                [this.name, this.type, this.doc_string] = PropertyCell{:};
                % Call constructors on properties that are Proto objects
                if ~isempty(this.type)
                    this.type = TypeProto(this.type);
                end
            end
        end
        
        function encodeValueInfoProto(this, CPtr)
            % Recursively fill the CPtr from 'this'.
            import nnet.internal.cnn.onnx.*
            PropertyCell = {this.name, this.type, this.doc_string};
            PtrCell = onnxmex(int32(FuncName.EencodeValueInfoProto), CPtr, PropertyCell);
            % Fill pointer objects
            arrayfun(@encodeTypeProto, this.type, PtrCell{2});
        end
    end
end