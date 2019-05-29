classdef NetworkMetadata
    % A class to hold ONNX network metadata
        
    % Copyright 2018-2019 The Mathworks, Inc.

    properties
        % Each comment gives the type that ONNX expects. MATLAB strings,
        % chars and any numeric type are allowed here, and will be
        % converted to the exact ONNX datatypes elsewhere when creating the
        % ModelProto. Empty values of any type will not be written to the
        % PB file.
        NetworkName     = "network";        % string
        IrVersion       = 3;                % integer
        OpsetVersion    = 6;                % integer
        ProducerName    = string(getString(message('nnet_cnn_onnx:onnx:ProducerName')));  % string
        ProducerVersion = "19.1.2";         % string. 
        Domain          = "";               % string
        ModelVersion    = [];               % integer
        DocString       = "";               % string
        % Mathworks metadata fields
        MathWorksOpsetVersion = 1;          % integer
    end
end