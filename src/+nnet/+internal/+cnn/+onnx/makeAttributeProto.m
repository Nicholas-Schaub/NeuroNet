function attributeProto = makeAttributeProto(name, Type, value)
% name is a char array.
% type is a char array.
% value is a type corresponding to 'type'
import nnet.internal.cnn.onnx.*
assert(ischar(name));
assert(ismember(upper(Type), {'FLOAT', 'INT', 'STRING', 'TENSOR', 'GRAPH', ...
    'FLOATS', 'INTS', 'STRINGS', 'TENSORS', 'GRAPHS'}));

attributeProto      = AttributeProto;
attributeProto.name = name;
attributeProto.type = AttributeProto_AttributeType.(upper(Type));
switch attributeProto.type
    case AttributeProto_AttributeType.FLOAT
        assert(isnumeric(value) && isscalar(value));
        attributeProto.f = single(value);
    case AttributeProto_AttributeType.INT
        assert(isnumeric(value) && isscalar(value));
        attributeProto.i = int64(value);
    case AttributeProto_AttributeType.STRING
        assert(ischar(value));
        attributeProto.s = value;
    case AttributeProto_AttributeType.TENSOR
        assert(isa(value, 'nnet.internal.cnn.onnx.TensorProto'));
        attributeProto.t = value;
    case AttributeProto_AttributeType.GRAPH
        assert(isa(value, 'nnet.internal.cnn.onnx.GraphProto'));
        attributeProto.g = value;
    case AttributeProto_AttributeType.FLOATS
        assert(isnumeric(value));
        attributeProto.floats = single(value);
    case AttributeProto_AttributeType.INTS
        assert(isnumeric(value));
        attributeProto.ints = int64(value);
    case AttributeProto_AttributeType.STRINGS
        assert(iscellstr(value));
        attributeProto.strings = value;
    case AttributeProto_AttributeType.TENSORS
        assert(isa(value, 'nnet.internal.cnn.onnx.TensorProto'));
        attributeProto.tensors = value;
    case AttributeProto_AttributeType.GRAPHS
        assert(isa(value, 'nnet.internal.cnn.onnx.GraphProto'));
        attributeProto.graphs = value;
    otherwise
        assert(0, 'type is not a supported member of nnet.internal.cnn.onnx.AttributeProto_AttributeType');
end
end