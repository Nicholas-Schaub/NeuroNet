classdef OperatorSetIdProto
    %   optional string domain = 1;
    %   optional int64 version = 2;
    properties
        domain
        version
    end
    
    methods
        function this = OperatorSetIdProto(varargin)
            import nnet.internal.cnn.onnx.*
            if nargin > 0
                Ptr = varargin{1};
                PropertyCell = onnxmex(int32(FuncName.EdecodeOperatorSetIdProto), Ptr);
                [this.domain, this.version] = PropertyCell{:};
                % Call constructors on pointer properties
                % (none)
            end
        end
        
        function encodeOperatorSetIdProto(this, CPtr)
            import nnet.internal.cnn.onnx.*
            % Create the C obj
            PropertyCell = {this.domain, this.version};
            PtrCell = onnxmex(int32(FuncName.EencodeOperatorSetIdProto), CPtr, PropertyCell);
            % Fill pointer objects
            % (none)
        end
    end
end

