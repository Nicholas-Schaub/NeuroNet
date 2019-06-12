function modelProto = decodeONNXFile(filename)
    
%   Copyright 2018 The MathWorks, Inc.

Filename = iValidateFile(filename);
modelProto = nnet.internal.cnn.onnx.ModelProto(Filename);
end

function Filepath = iValidateFile(Filename)
if ~(isa(Filename,'char') || isa(Filename,'string'))
    error('First arg must be a string.');
end
Filepath = which(char(Filename));
if isempty(Filepath) && exist(Filename, 'file')
    Filepath = Filename;
end
if ~exist(Filepath, 'file')
    error('File not found.');
end
end
