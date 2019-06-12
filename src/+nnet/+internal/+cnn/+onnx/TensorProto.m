classdef TensorProto
    %   repeated int64 dims = 1;
    %   optional DataType data_type = 2;
    %   optional Segment segment = 3;
    %   repeated float float_data = 4 [packed = true];
    %   repeated int32 int32_data = 5 [packed = true];
    %   repeated bytes string_data = 6;
    %   repeated int64 int64_data = 7 [packed = true];
    %   optional string name = 8;
    %   optional string doc_string = 12;
    %   optional bytes raw_data = 9;
    %   repeated double double_data = 10 [packed = true];
    %   repeated uint64 uint64_data = 11 [packed = true];
    
    % Data storage ordering: Line 308 of the ONNX protobuf definition file
    % (https://github.com/onnx/onnx/blob/master/onnx/onnx.proto) states
    % that "Tensor content must be in the row major order."

    properties
        dims
        data_type
        segment
        float_data
        int32_data
        string_data
        int64_data
        name
        doc_string
        raw_data
        double_data
        uint64_data
    end
    
    methods
        function this = TensorProto(varargin)
            import nnet.internal.cnn.onnx.*
            if nargin > 0
                Ptr = varargin{1};
                % Get raw properties
                PropertyCell = onnxmex(int32(FuncName.EdecodeTensorProto), Ptr);
                [this.dims, this.data_type, this.segment, this.float_data, this.int32_data, ...
                    this.string_data, this.int64_data, this.name, this.doc_string,...
                    this.raw_data, this.double_data, this.uint64_data] = PropertyCell{:};
                % Call constructors on properties that are pointer objects
                if ~isempty(this.data_type)
                    this.data_type = TensorProto_DataType(this.data_type);
                end
                if ~isempty(this.segment)
                    this.segment = TensorProto_Segment(this.segment);
                end
            end
        end
        
        function encodeTensorProto(this, CPtr)
            % Recursively fill the CPtr from 'this'.
            import nnet.internal.cnn.onnx.*
            PropertyCell = {this.dims, int32(this.data_type), this.segment, this.float_data, this.int32_data, ...
                this.string_data, this.int64_data, this.name, this.doc_string,...
                this.raw_data, this.double_data, this.uint64_data};
            PtrCell = onnxmex(int32(FuncName.EencodeTensorProto), CPtr, PropertyCell);
            % Fill pointer objects
            arrayfun(@encodeTensorProto_Segment, this.segment,      PtrCell{3});
        end
        
        % Type-checking setters
        function this = set.dims(this, val)
            assert(isempty(val) || isa(val,'int64'));
            this.dims = val;
        end
        
        function this = set.data_type(this, val)
            assert(isempty(val) || isa(val,'nnet.internal.cnn.onnx.TensorProto_DataType') || isa(val,'int32'));
            this.data_type = val;
        end
        
        function this = set.segment(this, val)
            assert(isempty(val) || isa(val,'nnet.internal.cnn.onnx.TensorProto_Segment'));
            this.segment = val;
        end
        
        function this = set.float_data(this, val)
            assert(isempty(val) || isa(val,'single'));
            this.float_data = val;
        end
        
        function this = set.int32_data(this, val)
            assert(isempty(val) || isa(val,'int32'));
            this.int32_data = val;
        end
        
        function this = set.string_data(this, val)
            assert(isempty(val) || isa(val,'uint8'));
            this.string_data = val;
        end
        
        function this = set.int64_data(this, val)
            assert(isempty(val) || isa(val,'int64'));
            this.int64_data = val;
        end
        
        function this = set.name(this, val)
            assert(isempty(val) || ischar(val));
            this.name = val;
        end
        
        function this = set.doc_string(this, val)
            assert(isempty(val) || ischar(val));
            this.doc_string = val;
        end
        
        function this = set.raw_data(this, val)
            assert(isempty(val) || isa(val,'uint8'));
            this.raw_data = val;
        end
        
        function this = set.double_data(this, val)
            assert(isempty(val) || isa(val,'double'));
            this.double_data = val;
        end
        
        function this = set.uint64_data(this, val)
            assert(isempty(val) || isa(val,'uint64'));
            this.uint64_data = val;
        end
    end
end
