classdef GraphProto
    %   repeated NodeProto node = 1;
    %   optional string name = 2;   // namespace Graph
    %   repeated TensorProto initializer = 5;
    %   optional string doc_string = 10;
    %   repeated ValueInfoProto input = 11;
    %   repeated ValueInfoProto output = 12;
    %   repeated ValueInfoProto value_info = 13;
    properties
        node
        name
        initializer
        doc_string
        input
        output
        value_info
    end
    
    methods
        function this = GraphProto(varargin)
            import nnet.internal.cnn.onnx.*
            if nargin > 0
                Ptr = varargin{1};
                % Get raw properties
                PropertyCell = onnxmex(int32(FuncName.EdecodeGraphProto), Ptr);
                [this.node, this.name, this.initializer, this.doc_string, this.input, ...
                    this.output, this.value_info] = PropertyCell{:};
                % Call constructors on properties that are Proto objects
                this.node           = arrayfun(@NodeProto, this.node);
                this.initializer	= arrayfun(@TensorProto, this.initializer);
                this.input          = arrayfun(@ValueInfoProto, this.input);
                this.output         = arrayfun(@ValueInfoProto, this.output);
                this.value_info     = arrayfun(@ValueInfoProto, this.value_info);
            end
        end
        
        function encodeGraphProto(this, CPtr)
            % Recursively fill the CPtr from 'this'.
            import nnet.internal.cnn.onnx.*
            PropertyCell = {this.node, this.name, this.initializer, this.doc_string, this.input, ...
                this.output, this.value_info};
            PtrCell = onnxmex(int32(FuncName.EencodeGraphProto), CPtr, PropertyCell);
            % Fill pointer objects
            arrayfun(@encodeNodeProto, this.node,               PtrCell{1});
            arrayfun(@encodeTensorProto, this.initializer,      PtrCell{3});
            arrayfun(@encodeValueInfoProto, this.input,         PtrCell{5});
            arrayfun(@encodeValueInfoProto, this.output,        PtrCell{6});
            arrayfun(@encodeValueInfoProto, this.value_info,    PtrCell{7});
        end
    end
end
