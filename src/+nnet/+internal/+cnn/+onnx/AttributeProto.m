classdef AttributeProto
    %   optional string name = 1;           // namespace Attribute
    %   optional string doc_string = 13;
    %   optional AttributeType type = 20;   // discriminator that indicates which field below is in use
    %   optional float f = 2;               // float
    %   optional int64 i = 3;               // int
    %   optional bytes s = 4;               // UTF-8 string
    %   optional TensorProto t = 5;         // tensor value
    %   optional GraphProto g = 6;          // graph
    %   repeated float floats = 7;          // list of floats
    %   repeated int64 ints = 8;            // list of ints
    %   repeated bytes strings = 9;         // list of UTF-8 strings
    %   repeated TensorProto tensors = 10;  // list of tensors
    %   repeated GraphProto graphs = 11;    // list of graph
    properties
        name
        doc_string
        type
        f
        i
        s
        t
        g
        floats
        ints
        strings
        tensors
        graphs
    end
    
    methods
        function this = AttributeProto(varargin)
            import nnet.internal.cnn.onnx.*
            if nargin > 0
                Ptr = varargin{1};
                % Get raw properties
                PropertyCell = onnxmex(int32(FuncName.EdecodeAttributeProto), Ptr);
                [this.name, this.doc_string, this.type, this.f, this.i, ...
                    this.s, this.t, this.g, this.floats, this.ints, this.strings, ...
                    this.tensors, this.graphs] = PropertyCell{:};
                % Call constructors on properties that are Proto objects
                if ~isempty(this.type)
                    this.type = AttributeProto_AttributeType(this.type);
                end
                if ~isempty(this.t)
                    this.t = TensorProto(this.t);
                end
                if ~isempty(this.g)
                    this.g = GraphProto(this.g);
                end
                this.tensors = arrayfun(@TensorProto, this.tensors);
                this.graphs  = arrayfun(@GraphProto, this.graphs);
            end
        end
        
        function encodeAttributeProto(this, CPtr)
            % Recursively fill the CPtr from 'this'.
            import nnet.internal.cnn.onnx.*
            PropertyCell = {this.name, this.doc_string, int32(this.type), this.f, this.i, ...
                this.s, this.t, this.g, this.floats, this.ints, this.strings, ...
                this.tensors, this.graphs};
            PtrCell = onnxmex(int32(FuncName.EencodeAttributeProto), CPtr, PropertyCell);
            % Fill pointer objects
            arrayfun(@encodeTensorProto, this.t,       PtrCell{7});
            arrayfun(@encodeGraphProto, this.g,        PtrCell{8});
            arrayfun(@encodeTensorProto, this.tensors, PtrCell{12});
            arrayfun(@encodeGraphProto, this.graphs,   PtrCell{13});
        end
    end
end