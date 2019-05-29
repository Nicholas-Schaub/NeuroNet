classdef NodeProto
    
    % message NodeProto {
    %   repeated string input = 1;    // namespace Value
    %   repeated string output = 2;   // namespace Value
    %   optional string name = 3;     // namespace Node
    %   optional string op_type = 4;  // namespace Operator
    %   optional string domain = 7;   // namespace Domain
    %   repeated AttributeProto attribute = 5;
    %   optional string doc_string = 6;
    properties
        input
        output
        name
        op_type
        domain
        attribute
        doc_string
    end
    
    methods
        function this = NodeProto(varargin)
            import nnet.internal.cnn.onnx.*
            if nargin > 0
                Ptr = varargin{1};
                % Get raw properties
                PropertyCell = onnxmex(int32(FuncName.EdecodeNodeProto), Ptr);
                [this.input, this.output, this.name, this.op_type, this.domain, ...
                    this.attribute, this.doc_string] = PropertyCell{:};
                % Call constructors on properties that are Proto objects
                this.attribute = arrayfun(@AttributeProto, this.attribute);
            end
        end
        
        function encodeNodeProto(this, CPtr)
            % Recursively fill the CPtr from 'this'.
            import nnet.internal.cnn.onnx.*
            PropertyCell = {this.input, this.output, this.name, this.op_type, this.domain, ...
                this.attribute, this.doc_string};
            PtrCell = onnxmex(int32(FuncName.EencodeNodeProto), CPtr, PropertyCell);
            % Fill pointer objects
            arrayfun(@encodeAttributeProto, this.attribute, PtrCell{6});
        end
        
        % Type-checking setters
        function this = set.input(this, val)
            assert(isempty(val) || iscellstr(val));
            this.input = val;
        end
        
        function this = set.output(this, val)
            assert(isempty(val) || iscellstr(val));
            this.output = val;
        end
        
        function this = set.name(this, val)
            assert(isempty(val) || ischar(val));
            this.name = val;
        end
        
        function this = set.op_type(this, val)
            assert(isempty(val) || ischar(val));
            this.op_type = val;
        end
        
        function this = set.domain(this, val)
            assert(isempty(val) || ischar(val));
            this.domain = val;
        end
        
        function this = set.doc_string(this, val)
            assert(isempty(val) || ischar(val));
            this.doc_string = val;
        end
    end
end