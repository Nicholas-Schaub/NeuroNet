function exportONNXNetwork(Network, filename, varargin)
%exportONNXNetwork  Export network to ONNX model format.
%
% exportONNXNetwork(net,filename) exports the deep learning network net
% with weights to the ONNX format file specified by filename. If filename
% exists, then exportONNXNetwork overwrites the file.
%
%  Inputs:
%  -------
%
%  net          - Trained network. A SeriesNetwork or DAGNetwork object.
%
%  filename     - Name of file. A character vector or string.
%
% exportONNXNetwork(net,filename, 'PARAM1', VAL1, 'PARAM2', VAL2, ...)
%  specifies additional name-value pairs described below:
%
%  'NetworkName'    - A string or character array specifying a name to
%                     store in the ONNX network.
%                     Default: 'Network'
%
%  'OpsetVersion'   - An integer specifying the version of the ONNX
%                     operator set to use. Supported versions are 6, 7, 8,
%                     9. Default: 6

% Copyright 2018-2019 The Mathworks, Inc.

%% Check if support package is installed
breadcrumbFile = 'nnet.internal.cnn.supportpackages.isOnnxInstalled';
fullpath = which(breadcrumbFile);
if isempty(fullpath)
    % Not installed; throw an error
    name = 'Deep Learning Toolbox Converter for ONNX Model Format';
    basecode = 'ONNXCONVERTER';
    error(message('nnet_cnn:supportpackages:NotInstalled', ...
        mfilename, name, basecode));
end

% Call the main function
nnet.internal.cnn.onnx.exportONNXNetwork(Network, filename, varargin{:});
end
