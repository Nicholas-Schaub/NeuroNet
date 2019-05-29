function LayersOrGraph = importONNXLayers(filename, varargin)

% Copyright 2018 The Mathworks, Inc.
nnet.internal.cnn.onnx.setAdditionalResourceLocation(); % For SPKG resource catalog.
[Filename, ImportWeights, OutputLayerType, UserImageInputSize] = iValidateInputs(filename, varargin{:});
% Import
modelProto    = nnet.internal.cnn.onnx.ModelProto(Filename);
LayersOrGraph = nnet.internal.cnn.onnx.translateONNX(modelProto, OutputLayerType, UserImageInputSize, ImportWeights);
LayersOrGraph = LayersOrGraph(:);           % Make it a column (in case its an array)
% Warn if there are unsupported layers
[~,Indices] = findPlaceholderLayers(LayersOrGraph);
if any(Indices)
    warning(message('nnet_cnn_onnx:onnx:UnsupportedLayerWarning'));
end
end

function [Filename, ImportWeights,OutputLayerType, ImageInputSize] = iValidateInputs(filename, varargin)
par = inputParser();
par.addParameter('OutputLayerType', '');
par.addParameter('ImportWeights', false);
% par.addParameter('ImageInputSize', []);
par.parse(varargin{:});

OutputLayerType = par.Results.OutputLayerType;
% ImageInputSize = par.Results.ImageInputSize;
ImageInputSize = [];
Filename = iValidateFile(filename);
OutputLayerType = iValidateOutputLayerType(OutputLayerType);
% ImageInputSize = iValidateImageInputSize(ImageInputSize);
ImportWeights = par.Results.ImportWeights;
iValidateImportWeights(ImportWeights);
end

function OutputLayerType = iValidateOutputLayerType(OutputLayerType)
if isempty(OutputLayerType)
    warning(message('nnet_cnn_onnx:onnx:OutputLayerTypeOptional'));
else
    OutputLayerType = validatestring(OutputLayerType, {'classification', 'regression', 'pixelclassification'}, ...
        'importONNXLayers', 'OutputLayerType');
    if isequal(OutputLayerType,'pixelclassification') && ~nnet.internal.cnn.onnx.isInstalledCVST
        throwAsCaller(MException(message('nnet_cnn_onnx:onnx:noCVSTForPixelClassification')));
    end
end
end

function ImageInputSize = iValidateImageInputSize(ImageInputSize)
if ~isempty(ImageInputSize) && ~(isreal(ImageInputSize) && isvector(ImageInputSize) && ...
        ismember(numel(ImageInputSize),[2 3]) && ...
        isequal(ImageInputSize, floor(ImageInputSize)) && ...
        all(ImageInputSize > 0))
    throwAsCaller(MException(message('nnet_cnn_onnx:onnx:ImageInputSizeBad')));
end
ImageInputSize = double(ImageInputSize(:));
end

function Filepath = iValidateFile(Filename)
if ~(isa(Filename,'char') || isa(Filename,'string'))
    throwAsCaller(MException(message('nnet_cnn_onnx:onnx:FirstArgString')));
end
Filepath = which(char(Filename));
if isempty(Filepath) && exist(Filename, 'file')
    Filepath = Filename;
end
if ~exist(Filepath, 'file')
    throwAsCaller(MException(message('nnet_cnn_onnx:onnx:FileNotFound', Filename)));
end
end

function iValidateImportWeights(ImportWeights)
if ~(isscalar(ImportWeights) && islogical(ImportWeights) || isequal(ImportWeights,0) || isequal(ImportWeights,1))
    throwAsCaller(MException(message('nnet_cnn_onnx:onnx:BadImportWeights')));
end
end

function tf = isUnsupportedLayer(Layer)
tf = isa(Layer, 'nnet.onnx.layer.PlaceholderLayer');
end