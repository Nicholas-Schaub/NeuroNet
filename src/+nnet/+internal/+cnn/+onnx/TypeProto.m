classdef TypeProto
    %   oneof value {
    %     Tensor tensor_type = 1;
    %     Sequence sequence_type = 4;
    %     Map map_type = 5;
    %   }
    properties
        tensor_type
        sequence_type
        map_type
    end
    
    methods
        function this = TypeProto(varargin)
            import nnet.internal.cnn.onnx.*
            if nargin > 0
                Ptr = varargin{1};
                % Get raw properties
                PropertyCell = onnxmex(int32(FuncName.EdecodeTypeProto), Ptr);
                [this.tensor_type, this.sequence_type, this.map_type] = PropertyCell{:};
                % Call constructors on properties that are Proto objects
                if ~isempty(this.tensor_type)
                    this.tensor_type = TypeProto_Tensor(this.tensor_type);
                end
                if ~isempty(this.sequence_type)
                    this.sequence_type = TypeProto_Sequence(this.sequence_type);
                end
                if ~isempty(this.map_type)
                    this.map_type = TypeProto_Map(this.map_type);
                end
            end
        end
        
        function encodeTypeProto(this, CPtr)
            % Recursively fill the CPtr from 'this'.
            import nnet.internal.cnn.onnx.*
            PropertyCell = {this.tensor_type, this.sequence_type, this.map_type};
            PtrCell = onnxmex(int32(FuncName.EencodeTypeProto), CPtr, PropertyCell);
            % Fill pointer objects
            arrayfun(@encodeTypeProto_Tensor, this.tensor_type,     PtrCell{1});
            arrayfun(@encodeTypeProto_Sequence, this.sequence_type, PtrCell{2});
            arrayfun(@encodeTypeProto_Map, this.map_type,           PtrCell{3});
        end
    end
end
