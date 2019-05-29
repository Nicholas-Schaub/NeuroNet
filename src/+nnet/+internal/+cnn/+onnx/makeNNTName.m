function NNTName = makeNNTName(ONNXLayerName)
% Make the ONNX layer name compatible with NNT. 
% * Change forward slashes to vertical bars
% ONNXLayerName is a char vector.
NNTName = strrep(ONNXLayerName, '/', '|');
end