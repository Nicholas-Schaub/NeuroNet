function layer = translateLSTM(node,LayerName,initializerDimMap,initializerRawDataMap,ImportWeights, OpsetVersion)

%   Copyright 2018 The MathWorks, Inc.

%Get the attributes
attributeNames = arrayfun(@(a) a.name, node.attribute,'UniformOutput',false);
attributeInt = arrayfun(@(a) a.i, node.attribute,'UniformOutput',false);
attributeString = arrayfun(@(a) a.s, node.attribute,'UniformOutput',false);
attributeStrings = arrayfun(@(a) a.strings, node.attribute,'UniformOutput',false);
attributeIntMap = containers.Map(attributeNames, attributeInt);
attributeStringMap = containers.Map(attributeNames, attributeString);
attributeStringsMap = containers.Map(attributeNames, attributeStrings);

hiddenSize = attributeIntMap('hidden_size');
direction = attributeStringMap('direction');
activations = attributeStringsMap('activations');

iofAct = iTranslateActivationFunction(activations{1});
cellAct = iTranslateActivationFunction(activations{2});
hidAct = iTranslateActivationFunction(activations{3});

% Check that activation functions are valid
if ~ismember(iofAct, {'sigmoid', 'hard-sigmoid'})
    warning(message('nnet_cnn_onnx:onnx:UnsupportedGateActivation', iofAct));
    layer = [];
    return;
end
if ~ismember(cellAct, {'tanh', 'softsign'})
    warning(message('nnet_cnn_onnx:onnx:UnsupportedCellActivation', cellAct));
    layer = [];
    return;
end
if ~ismember(hidAct, {'tanh', 'softsign'})
    warning(message('nnet_cnn_onnx:onnx:UnsupportedHiddenActivation', hidAct));
    layer = [];
    return;
end
if ~isequal(hidAct, cellAct)
    warning(message('nnet_cnn_onnx:onnx:CellAndHiddenActsUnequal', cellAct, hidAct));
    layer = [];
    return;
end

% Determine OutputMode:
if numel(node.output)>=1 && ~isempty(node.output{1})
    OutputMode = 'sequence';
elseif numel(node.output)>=2 && ~isempty(node.output{2})
    OutputMode = 'last';
else
    warning(message('nnet_cnn_onnx:onnx:BadLSTMOutput', LayerName));
    layer = [];
    return;
end

% Create DLT layer
if strcmp(direction, 'forward')
    layer = lstmLayer(hiddenSize, 'Name', LayerName, ...
        'OutputMode', OutputMode,...
        'StateActivationFunction', hidAct,...
        'GateActivationFunction', iofAct);
elseif strcmp(direction, 'bidirectional')
    layer = bilstmLayer(hiddenSize, 'Name', LayerName, ...
        'OutputMode', OutputMode,...
        'StateActivationFunction', hidAct,...
        'GateActivationFunction', iofAct);
else
    warning(message('nnet_cnn_onnx:onnx:ReverseLSTM'));
    layer = [];
    return;
end

if ImportWeights
    % input weights
    inputWeightName = node.input{2};
    W = single(initializerRawDataMap(inputWeightName));     % ONNX shape is [num_directions, 4*hidden_size, input_size], row-major.
    W = iFixWeightsShape(W, hiddenSize, direction);
    layer.InputWeights = W;
    % recurrent weights
    recurrentWeightName = node.input{3};
    R = single(initializerRawDataMap(recurrentWeightName)); % ONNX shape is [num_directions, 4*hidden_size, hidden_size], row-major.
    R = iFixWeightsShape(R, hiddenSize, direction);
    layer.RecurrentWeights = R;
    % biases: optional in ONNX
    if numel(node.input) >= 4 && ~isempty(node.input{4})
        biasName = node.input{4};
        B = single(initializerRawDataMap(biasName));        % ONNX shape is [num_directions, 8*hidden_size], row-major.
        B = iFixBiasShape(B, hiddenSize, direction);
    else
        B = zeros(4*hiddenSize,1,'single');
    end
    % initial_h: optional in ONNX
    if numel(node.input) >= 6 && ~isempty(node.input{6})
        initialhName = node.input{6};
        H = single(initializerRawDataMap(initialhName));    % ONNX shape is [num_directions, batch_size, hidden_size], row-major.
        H = iFixInitialhShape(H, hiddenSize, direction);
    else
        H = zeros(2*hiddenSize,1,'single');
    end
    % initial_c: optional in ONNX
    if numel(node.input) >= 7 && ~isempty(node.input{7})
        initialcName = node.input{7};
        C = single(initializerRawDataMap(initialcName));    % ONNX shape is [num_directions, batch_size, hidden_size], row-major.
        C = iFixInitialhShape(C, hiddenSize, direction);
    else
        C = zeros(2*hiddenSize,1,'single');
    end
    layer.Bias        = B;
    layer.HiddenState = H;
    layer.CellState   = C;
end
end

function H = iFixInitialhShape(H, hiddenSize, direction)
% ONNX shape is [num_directions, batch_size, hidden_size], row-major.
% MATLAB order is [hidden_size,1] if forward, [2*hidden_size,1] if bidirectional
switch direction
    case 'forward'
        H = reshape(H, [hiddenSize, 1, 1]);                     % Now H is col-major.
    case 'bidirectional'
        H = reshape(H, [hiddenSize, 1, 2]);                     % Now H is col-major.
        H = reshape(H, [2*hiddenSize, 1]);
    otherwise
        assert(false);
end
end

function B = iFixBiasShape(B, hiddenSize, direction)
% ONNX order is Input-Output-Forget-Cell (x2 if bidirectional)
% MATLAB order is Input-Forget-Cell-Output (x2 if bidirectional)
iofcRows = reshape(1:4*hiddenSize, [], 4);                      % Columns of this correspond to ONNX's iofc ordering.
ifcoRows = iofcRows(:, [1 3 4 2]);                              % Columns of this correspond to DLT's ifco ordering.
ifcoInd  = ifcoRows(:);                                         % A column vector containing DLT's ifco ordering.
switch direction
    case 'forward'
        % B is [1, 8*hidden_size], row-major. (Same col-major)
        B(1:4*hiddenSize)   = B(ifcoInd);                       % First half of B is now in DLT's required ifco order.
        B                   = B(:);
    case 'bidirectional'
        % B is [2, 8*hidden_size], row-major.
        B                   = reshape(B, [8*hiddenSize, 2]);    % B is now [8*hiddenSize, 2] col-major (using the "fliplr theorem").
        B(1:4*hiddenSize,:)	= B(ifcoInd,:);                     % First half of each column of B is now in DLT's required ifco order.
    otherwise
        assert(false);
end
% MATLAB does not support recurrent biases. These would appear as
% nonzeros in the second half of each column of B.
if any(B(4*hiddenSize+1:end, :),'all')
    error(message('nnet_cnn_onnx:onnx:RecurrentBiases'));
end
% Return the concatenated first half of each column of B
B = B(1:4*hiddenSize, :);
B = B(:);
end

function W = iFixWeightsShape(W, hiddenSize, direction)
% ONNX order is Input-Output-Forget-Cell (x2 if bidirectional)
% MATLAB order is Input-Forget-Cell-Output (x2 if bidirectional)
iofcRows = reshape(1:4*hiddenSize, [], 4);                      % Columns of this correspond to ONNX's iofc ordering.
ifcoRows = iofcRows(:, [1 3 4 2]);                              % Columns of this correspond to DLT's ifco ordering.
ifcoInd  = ifcoRows(:);                                         % A column vector containing DLT's ifco ordering.
switch direction
    case 'forward'
        inputSize   = numel(W)/(4*hiddenSize);                  % W is [1, 4*hidden_size, input_size], row-major.
        W           = reshape(W, [inputSize, 4*hiddenSize]);    % W is now [inputSize, 4*hiddenSize] col-major (using the "fliplr theorem").
        W           = W';                                       % W is now [4*hiddenSize, inputSize].
        W           = W(ifcoInd, :);                            % Rows of W are now in DLT's required ifco order.
    case 'bidirectional'
        inputSize   = numel(W)/(8*hiddenSize);                  % W is [2, 4*hidden_size, input_size], row-major.
        W           = reshape(W, [inputSize, 4*hiddenSize, 2]); % W is now [inputSize, 4*hiddenSize, 2] col-major (using the "fliplr theorem").
        W           = reshape(W, [inputSize, 8*hiddenSize]);    % W is now [inputSize, 8*hiddenSize].
        W           = W';                                       % W is now [8*hiddenSize, inputSize].
        W           = W([ifcoInd; ifcoInd+4*hiddenSize], :);    % Rows of W are now in DLT's required ifcoifco order.
    otherwise
        assert(false);
end
end

function s = iTranslateActivationFunction(s)
switch s
    case 'Sigmoid'
        s = 'sigmoid';
    case 'Softsign'
        s = 'softsign';
    case 'Tanh'
        s = 'tanh';
    case 'HardSigmoid'
        s = 'hard-sigmoid';
    otherwise
        % Maintain name for error message
end
end

