function vip = makeValueInfoProtoFromTensorProto(tp)
% Make a ValueInfoProto that describes the TensorProto tp.
import nnet.internal.cnn.onnx.*
vip = makeValueInfoProtoFromDimensions(tp.name, tp.data_type, tp.dims);
end