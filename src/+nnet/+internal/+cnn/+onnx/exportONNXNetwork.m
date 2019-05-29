function exportONNXNetwork(NNTNetwork, Filename, varargin)
%exportONNXNetwork  Export a network and weights to an ONNX format protobuf file.
%
% exportONNXNetwork(Network, filename) exports a network and its weights
% to filename.
%
%  Inputs:
%  -------
%
%  Network      - A DAGNetwork or SeriesNetwork.
%
%  filename   	- A string or character array.
%
%  exportONNXNetwork(..., Name, Value) specifies additional name-value
%  pairs described below:
%
%  'NetworkName'    - A string or character array specifying a name to
%                     store in the ONNX network.
%  'OpsetVersion'   - In integer specifying the version of the ONNX
%                     operator set to use. Supported versions are 6, 7, 8,
%                     9. Default: 6

% Copyright 2018 The Mathworks, Inc.

nnet.internal.cnn.onnx.setAdditionalResourceLocation();     % For SPKG resource catalog.
% Check input
[NNTNetwork, Filename, NetworkName, OpsetVersion] = iValidateInputs(NNTNetwork, Filename, varargin{:});
% Update metadata
metadata = nnet.internal.cnn.onnx.NetworkMetadata;
metadata.NetworkName = NetworkName;
metadata.OpsetVersion = OpsetVersion;
% Convert
converter   = nnet.internal.cnn.onnx.ConverterForNetwork(NNTNetwork, metadata);
modelProto  = toOnnx(converter);
% Write
writeToFile(modelProto, Filename);
end

function [NNTNetwork, Filename, NetworkName, OpsetVersion] = iValidateInputs(NNTNetwork, Filename, varargin)
% Setup parser
par = inputParser();
par.addRequired('NNTNetwork');
par.addRequired('Filename');
par.addParameter('NetworkName', "Network");
par.addParameter('OpsetVersion', 6);
% Parse
par.parse(NNTNetwork, Filename, varargin{:});
NetworkName = par.Results.NetworkName;
OpsetVersion = par.Results.OpsetVersion;
% Validate
NNTNetwork  = iValidateNetwork(NNTNetwork);
Filename    = iValidateFilename(Filename);
NetworkName = iValidateNetworkName(NetworkName);
OpsetVersion = iValidateOpsetVersion(OpsetVersion);
end

function Network = iValidateNetwork(Network)
if ~(isa(Network, 'DAGNetwork') || isa(Network, 'SeriesNetwork'))
    error(message('nnet_cnn_onnx:onnx:NetworkWrongType'));
end
end

function Filename = iValidateFilename(Filename)
if ~(isstring(Filename) || ischar(Filename))
    error(message('nnet_cnn_onnx:onnx:FilenameWrongType'));
end
Filename = char(Filename);
end

function NetworkName = iValidateNetworkName(NetworkName)
if ~(isstring(NetworkName) || ischar(NetworkName))
    error(message('nnet_cnn_onnx:onnx:NetworkNameWrongType'));
end
NetworkName = char(NetworkName);
end

function OpsetVersion = iValidateOpsetVersion(OpsetVersion)
SupportedOpsetsForExport = [6,7,8,9];
if ~any(OpsetVersion == SupportedOpsetsForExport)
    error(message('nnet_cnn_onnx:onnx:OpsetVersionUnsupportedForExport', SupportedOpsetsForExport));
end
OpsetVersion = double(OpsetVersion);
end
