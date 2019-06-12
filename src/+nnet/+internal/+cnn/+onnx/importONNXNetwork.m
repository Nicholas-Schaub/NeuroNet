function Network = importONNXNetwork(filename,varargin)
    
%   Copyright 2018 The MathWorks, Inc.
nnet.internal.cnn.onnx.setAdditionalResourceLocation();     % For SPKG resource catalog.
[Filename, OutputLayerType, UserImageInputSize, Classes] = iValidateInputs(filename, varargin{:});
% Import
modelProto      = nnet.internal.cnn.onnx.ModelProto(Filename);
LayersOrGraph   = nnet.internal.cnn.onnx.translateONNX(modelProto, OutputLayerType, UserImageInputSize, true);
LayersOrGraph	= iConfigureOutputLayer(LayersOrGraph, Classes);
C               = iSuppressAutoClassesWarning();
Network         = assembleNetwork(LayersOrGraph);
end

function LayersOrGraph = iConfigureOutputLayer(LayersOrGraph, PassedClasses)
C = iSuppressAutoClassesWarning();
% Get layers
isLG = isa(LayersOrGraph, 'nnet.cnn.LayerGraph');
if isLG
    Layers = LayersOrGraph.Layers;
else
    Layers = LayersOrGraph;
end
% Check for RNN
isRNN = isa(Layers(1), 'nnet.cnn.layer.SequenceInputLayer');
% Error if any unsupported layers, refer user to importONNXLayers
[~,Indices] = findPlaceholderLayers(Layers);
if any(Indices)
    throwAsCaller(MException(message('nnet_cnn_onnx:onnx:CantBuildNetWithUnsupportedLayers')));
end
% Find output layer
locC = find(arrayfun(@(L)isa(L,'nnet.cnn.layer.ClassificationOutputLayer') || ...
                         isa(L,'nnet.cnn.layer.PixelClassificationLayer'), Layers));
locR = find(arrayfun(@(L)isa(L,'nnet.cnn.layer.RegressionOutputLayer'), Layers));
assert(numel(locC) + numel(locR) == 1);  % Must have exactly 1 output layer.
if ~isempty(locC)
    % It's a classification layer.
    % Analyze layers to determine whether it should be a pixelClassificationLayer
    na                    = nnet.internal.cnn.analyzer.NetworkAnalyzer(LayersOrGraph);
    naOut                 = na.LayerAnalyzers([na.LayerAnalyzers.IsOutputLayer]);
    outputTensorSize      = naOut.Inputs{'in','Size'}{1};
    isPixelClassification = isa(Layers(locC),'nnet.cnn.layer.PixelClassificationLayer');
else
    outputTensorSize = [];
    isPixelClassification = false;
end
iCheckOutputLayerTypeAndClasses(locR, PassedClasses, isRNN, outputTensorSize);
% Create a new output layer if needed
if ~iIsAuto(PassedClasses)
    if isPixelClassification
        newOutputLayer = pixelClassificationLayer('Name', naOut.Name, 'Classes', PassedClasses);
    else
        newOutputLayer = classificationLayer('Name', naOut.Name, 'Classes', PassedClasses);
    end
    % Put the new output layer in
    if isLG
        LayersOrGraph = replaceLayer(LayersOrGraph, newOutputLayer.Name, newOutputLayer);
    else
        LayersOrGraph(locC) = newOutputLayer;
    end
end
end
function iCheckOutputLayerTypeAndClasses(locR, PassedClasses, isRNN, outputTensorSize)
if ~isempty(locR) 
    if ~iIsAuto(PassedClasses)
    throwAsCaller(MException(message('nnet_cnn_onnx:onnx:ClassNamesForRegression')));
    end
elseif iIsAuto(PassedClasses)
    % 'auto' classes passed. assembleNetwork will set them later.
    warning(message('nnet_cnn_onnx:onnx:FillingInClassNames'));
else
    % Classes passed.
    % Verify that the user passed the right number.
    if isRNN
        netNumClasses = outputTensorSize(1);
    else
        netNumClasses = outputTensorSize(3);
    end
    iVerifyNumClasses(netNumClasses, PassedClasses);
end
end

function [Filename, OutputLayerType, ImageInputSize, Classes] = iValidateInputs(filename, varargin)
defaultClasses = 'auto';
defaultClassNames = {};
par = inputParser();
par.addParameter('OutputLayerType', '');
par.addParameter('ClassNames', defaultClassNames);
par.addParameter('Classes', defaultClasses, @iAssertValidClasses);
par.parse(varargin{:});
OutputLayerType = par.Results.OutputLayerType;
% ImageInputSize = par.Results.ImageInputSize;
ImageInputSize = [];
Filename = iValidateFile(filename);
OutputLayerType = iValidateOutputLayerType(OutputLayerType);
% ImageInputSize = iValidateImageInputSize(ImageInputSize);
ClassNames = par.Results.ClassNames;
Classes = par.Results.Classes;

if iIsSpecified(ClassNames, defaultClassNames) && ...
        iIsSpecified(Classes, defaultClasses)
    throwAsCaller(MException(message(...
        'nnet_cnn_onnx:onnx:ClassesAndClassNamesNVP')))
elseif iIsSpecified(ClassNames, defaultClassNames)
    ClassNames = iValidateClassNames(ClassNames);
    Classes = categorical(ClassNames, ClassNames);
elseif iIsSpecified(Classes, defaultClasses)
    Classes = iConvertClassesToCanonicalForm(Classes); 
else
    % Not specified ClassNames nor Classes. Do nothing.
end

% if ~isempty(ClassNames)
%     ClassNames = iValidateClassNames(ClassNames);
% end

end

function  ClassNames = iValidateClassNames(ClassNames)
if isstring(ClassNames)
    ClassNames = cellstr(ClassNames);
elseif ~iscellstr(ClassNames)
    throwAsCaller(MException(message('nnet_cnn_onnx:onnx:InvalidClassNames')));
end
if ~isvector(ClassNames)
    throwAsCaller(MException(message('nnet_cnn_onnx:onnx:InvalidClassNames')));
end
% make sure it's a column vector
ClassNames = ClassNames(:);
end

function iAssertValidClasses(value)
nnet.internal.cnn.layer.paramvalidation.validateClasses(value);
end

function classes = iConvertClassesToCanonicalForm(classes)
classes = ...
    nnet.internal.cnn.layer.paramvalidation.convertClassesToCanonicalForm(classes);
end

function OutputLayerType = iValidateOutputLayerType(OutputLayerType)
if isempty(OutputLayerType)
    throwAsCaller(MException(message('nnet_cnn_onnx:onnx:OutputLayerTypeMissing')));
end
OutputLayerType = validatestring(OutputLayerType, {'classification', 'regression', 'pixelclassification'}, ...
    'importONNXNetwork', 'OutputLayerType');
if isequal(OutputLayerType,'pixelclassification') && ~nnet.internal.cnn.onnx.isInstalledCVST
    throwAsCaller(MException(message('nnet_cnn_onnx:onnx:noCVSTForPixelClassification')));
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

function tf = iIsSpecified(value, defaultValue)
tf = ~isequal(convertStringsToChars(value), defaultValue);
end

function iVerifyNumClasses(NetNumClasses, PassedClasses)
if NetNumClasses ~= numel(PassedClasses)
    throwAsCaller(MException(message('nnet_cnn_onnx:onnx:ClassesMismatchSize', numel(NetNumClasses), numel(PassedClasses))));
end
end

function C = iSuppressAutoClassesWarning()
warnState = warning('off', 'nnet_cnn:internal:cnn:analyzer:NetworkAnalyzer:NetworkHasWarnings');
C = onCleanup(@()warning(warnState));
end

function tf = iIsAuto(val)
tf = isequal(string(val), "auto");
end