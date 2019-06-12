classdef ModelProto
    % Based on the protobuf ONNX message:
    % message ModelProto {
    %   optional int64 ir_version = 1;
    %   repeated OperatorSetIdProto opset_import = 8;
    %   optional string producer_name = 2;
    %   optional string producer_version = 3;
    %   optional string domain = 4;
    %   optional int64 model_version = 5;
    %   optional string doc_string = 6;
    %   optional GraphProto graph = 7;
    %   repeated StringStringEntryProto metadata_props = 14;
    properties
        ir_version
        opset_import
        producer_name
        producer_version
        domain
        model_version
        doc_string
        graph
        metadata_props
    end
    
    methods
        function this = ModelProto(varargin)
            import nnet.internal.cnn.onnx.*
            if nargin > 0
                filename = varargin{1};
                % Deserialize, creating the full C ModelProto.
                ModelPtr = onnxmex(int32(FuncName.EdeserializeFromFile), filename);
                C = onCleanup(@()onnxmex(int32(FuncName.EdestroyModelProto), ModelPtr)); % Prepare to deallocate the C ModelProto (recursively).
                % Get shallow properties
                PropertyCell = onnxmex(int32(FuncName.EdecodeModelProto), ModelPtr);
                [this.ir_version, this.opset_import, this.producer_name, this.producer_version, ...
                    this.domain, this.model_version, this.doc_string, this.graph, ...
                    this.metadata_props] = PropertyCell{:};
                % Call constructors on properties that are Proto objects
                this.opset_import = arrayfun(@OperatorSetIdProto, this.opset_import);
                if ~isempty(this.graph)
                    this.graph = GraphProto(this.graph);
                end
                this.metadata_props = arrayfun(@StringStringEntryProto, this.metadata_props);
            end
        end
        
        function writeToFile(this, filename)
            import nnet.internal.cnn.onnx.*
            % Allocate a shallow C ModelProto.
            ModelPtr = onnxmex(int32(FuncName.EnewModelProto));
            C = onCleanup(@()onnxmex(int32(FuncName.EdestroyModelProto), ModelPtr)); % Prepare to deallocate the C ModelProto (recursively).
            % Fill it recursively from MATLAB properties
            encodeModelProto(this, ModelPtr);
            % Serialize it to file.
            onnxmex(int32(FuncName.EserializeToFile), ModelPtr, filename);
        end
    end
    
    methods(Access=protected)
        function encodeModelProto(this, CPtr)
            % Recursively fill the CPtr from 'this'.
            import nnet.internal.cnn.onnx.*
            % Fill the ModelProto shallowly. Allocate sub-objects, returning pointers.
            PropertyCell = {this.ir_version, this.opset_import, this.producer_name, this.producer_version, ...
                this.domain, this.model_version, this.doc_string, this.graph, ...
                this.metadata_props};
            PtrCell = onnxmex(int32(FuncName.EencodeModelProto), CPtr, PropertyCell);
            % Fill pointer objects
            arrayfun(@encodeOperatorSetIdProto, this.opset_import,          PtrCell{2});
            arrayfun(@encodeGraphProto, this.graph,                         PtrCell{8});
            arrayfun(@encodeStringStringEntryProto, this.metadata_props,    PtrCell{9});
        end
    end
end

