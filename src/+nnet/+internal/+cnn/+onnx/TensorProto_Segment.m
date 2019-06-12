classdef TensorProto_Segment
    %     optional int64 begin = 1;
    %     optional int64 end = 2;
    properties
        begin
        end_    % Use underscore because 'end' is a reserved word.
    end
    
    methods
        function this = TensorProto_Segment(varargin)
            import nnet.internal.cnn.onnx.*
            if nargin > 0
                Ptr = varargin{1};
                % Get raw properties
                PropertyCell = onnxmex(int32(FuncName.EdecodeTensorProto_Segment), Ptr);
                [this.begin, this.end_] = PropertyCell{:};
                % Call constructors on properties that are Proto objects
                % (none)
            end
        end
        
        function encodeTensorProto_Segment(this, CPtr)
            % Recursively fill the CPtr from 'this'.
            import nnet.internal.cnn.onnx.*
            PropertyCell = {this.begin, this.end_};
            PtrCell = onnxmex(int32(FuncName.EencodeTensorProto_Segment), CPtr, PropertyCell);
        end
    end
end