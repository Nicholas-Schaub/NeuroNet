% classdef ConverterForBiLSTMLayer < nnet.internal.cnn.onnx.NNTLayerConverter
%     % Class to convert a bilstmLayer into ONNX
%     
%     % Copyright 2018 The Mathworks, Inc.
%     
%     methods
%         function this = ConverterForBiLSTMLayer(layerAnalyzer)
%             this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
%         end
%         
%         function [nodeProto, parameterInitializers, parameterInputs, networkInputs, networkOutputs, layerMap] = toOnnx(this, TensorNameMap)
%             import nnet.internal.cnn.onnx.*
%             
%             [lstmName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
%             NNTLayer = this.NNTLayer;
%             hasSequenceOutput = isequal(NNTLayer.OutputMode, 'sequence');
%             
%             %% (1) Make the LSTM node
%             nodeProto(1)            = NodeProto;
%             nodeProto(1).op_type    = 'LSTM';
%             nodeProto(1).name        = lstmName;
%             input                   = {...
%                 this.InputLayerNames{1},... % X
%                 [lstmName '_W'],...
%                 [lstmName '_R'],...
%                 [lstmName '_B'],...
%                 '',...                      % sequence_Lens
%                 [lstmName '_initial_h'],...
%                 [lstmName '_initial_c']
%                 };
%             nodeProto(1).input = mapTensorNames(this, input(:)', TensorNameMap);
%             % Make the output name appear as either the sequence output or
%             % the last output as appropriate.
%             if hasSequenceOutput
%                 nodeProto(1).output = {lstmName};          % Output is Y=onnxName
%             else
%                 nodeProto(1).output = {'', lstmName};      % Output is Y='', Y_h=onnxName
%             end
%             
%             % Set attributes
%             iofAct	= iTranslateActivationFunction(NNTLayer.GateActivationFunction);
%             cellAct	= iTranslateActivationFunction(NNTLayer.StateActivationFunction);
%             hidAct	= cellAct;
%             nodeProto(1).attribute = [...
%                 makeAttributeProto('activations',       'STRINGS',	{iofAct, cellAct, hidAct, iofAct, cellAct, hidAct}),...
%                 makeAttributeProto('direction',         'STRING',    'bidirectional'),...
%                 makeAttributeProto('hidden_size',       'INT',       NNTLayer.NumHiddenUnits),...
%                 ];
%             if this.OpsetVersion < 7
%                 nodeProto(1).attribute(end+1) = makeAttributeProto('output_sequence', 'INT', single(hasSequenceOutput));
%             end
%             
%             % Get weight indices
%             nH = NNTLayer.NumHiddenUnits;
%             [cInd, iInd, fInd, oInd] = nnet.internal.cnn.util.gateIndices(nH);
%             [forwardInd, backwardInd] = nnet.internal.cnn.util.forwardBackwardSequenceIndices(nH);
%             iofcForward = forwardInd([iInd oInd fInd cInd]);
%             iofcBackward = backwardInd([iInd oInd fInd cInd]);
%             
%             % Make parameterInitializers:
%             % W
%             inputWeights	= NNTLayer.InputWeights;
%             W               = inputWeights(iofcForward,:);
%             WB              = inputWeights(iofcBackward,:);
%             WTensor         = zeros(2, 4*nH, NNTLayer.InputSize, 'single');
%             WTensor(1,:,:)  = W;
%             WTensor(2,:,:)  = WB;
%             t1              = TensorProto;
%             t1.name         = [lstmName '_W'];
%             t1.data_type    = TensorProto_DataType.FLOAT;
%             t1.raw_data     = rawData(single(WTensor));
%             t1.dims         = dimVector(size(WTensor),3);
%             
%             % R
%             recurrentWeights = NNTLayer.RecurrentWeights;
%             W               = recurrentWeights(iofcForward,:);
%             WB              = recurrentWeights(iofcBackward,:);
%             RTensor         = zeros(2, 4*nH, nH, 'single');
%             RTensor(1,:,:)  = W;
%             RTensor(2,:,:)  = WB;
%             t2              = TensorProto;
%             t2.name         = [lstmName '_R'];
%             t2.data_type    = TensorProto_DataType.FLOAT;
%             t2.raw_data     = rawData(single(RTensor));
%             t2.dims         = dimVector(size(RTensor),3);
%             
%             % B
%             Bias            = NNTLayer.Bias;
%             B               = Bias(iofcForward);
%             RB          	= Bias(iofcBackward);
%             nB              = numel(B);
%             BTensor         = zeros(2, 8*nH, 'single');
%             BTensor(1,1:nB)	= B;                        % Forward "recurrent biases" are left at 0.
%             BTensor(2,1:nB)	= RB;                       % Backward "recurrent biases" are left at 0.
%             t3              = TensorProto;
%             t3.name         = [lstmName '_B'];
%             t3.data_type    = TensorProto_DataType.FLOAT;
%             t3.raw_data     = rawData(single(BTensor));
%             t3.dims         = dimVector(size(BTensor),2);
%             
%             % initial_h
%             HiddenState     = NNTLayer.HiddenState;                                         % NNT size is [2*nH, batchSize]
%             batchSize       = size(HiddenState,2);
%             initial_h       = permute(reshape(HiddenState, [2, nH, batchSize]), [1 3 2]);   % ONNX size is [2, batchSize, nH]
%             t4              = TensorProto;
%             t4.name         = [lstmName '_initial_h'];
%             t4.data_type    = TensorProto_DataType.FLOAT;
%             t4.raw_data     = rawData(single(initial_h));
%             t4.dims         = dimVector(size(initial_h),3);
%             
%             % initial_c
%             CellState       = NNTLayer.CellState;                                         % NNT size is [2*nH, batchSize]
%             batchSize       = size(CellState,2);
%             initial_c       = permute(reshape(CellState, [2, nH, batchSize]), [1 3 2]);   % ONNX size is [2, batchSize, nH]
%             t5              = TensorProto;
%             t5.name         = [lstmName '_initial_c'];
%             t5.data_type    = TensorProto_DataType.FLOAT;
%             t5.raw_data     = rawData(single(initial_c));
%             t5.dims         = dimVector(size(initial_c),3);
%             
%             finalNodeName         = lstmName;   % This may be changed below.
%             parameterInitializers = [t1 t2 t3 t4 t5];
%             
%             %%
%             % (2) For sequence output, Transpose and Reshape the output of
%             % the LSTM node to remove the 'num_directions' dimension.
%             if hasSequenceOutput
%                 transposeName           = [lstmName '_Transpose'];
%                 nodeProto(2)            = NodeProto;
%                 nodeProto(2).op_type    = 'Transpose';
%                 nodeProto(2).name       = transposeName;
%                 nodeProto(2).input      = {lstmName};
%                 nodeProto(2).output     = {transposeName};
%                 nodeProto(2).attribute	= makeAttributeProto('perm', 'INTS', [0 2 1 3]);
%                 % Reshape takes shape as an input. Make it an initializer.
%                 % First make a shape tensor
%                 shapeTensorName         = [lstmName '_Reshape_shape'];
%                 shapeTensor             = TensorProto;
%                 shapeTensor.name        = shapeTensorName;
%                 shapeTensor.data_type   = TensorProto_DataType.INT64;
%                 shapeTensor.raw_data    = rawData(int64([0 0 -1]));
%                 shapeTensor.dims      	= dimVector(3,1);
%                 % Then make the Reshape node
%                 reshapeName             = [lstmName '_Reshape'];
%                 nodeProto(3)            = NodeProto;
%                 nodeProto(3).op_type    = 'Reshape';
%                 nodeProto(3).name       = reshapeName;
%                 nodeProto(3).input      = {transposeName, shapeTensorName};
%                 nodeProto(3).output     = {reshapeName};
%                 
%                 parameterInitializers(end+1) = shapeTensor;
%                 finalNodeName               = reshapeName;
%             end
%             
%             %%
%             % Make parameterInputs
%             parameterInputs = arrayfun(@makeValueInfoProtoFromTensorProto, parameterInitializers);
%             
%             networkInputs           = [];
%             networkOutputs          = [];
%             layerMap                = containers.Map;
%             
%             if nameChanged
%                 layerMap(this.NNTLayer.Name) = finalNodeName;
%             end
%         end
%     end
% end
% 
% function s = iTranslateActivationFunction(s)
% switch s
%     case 'sigmoid'
%         s = 'Sigmoid';
%     case 'softsign'
%         s = 'Softsign';
%     case 'tanh'
%         s = 'Tanh';
%     case 'hard-sigmoid'
%         s = 'HardSigmoid';
%     otherwise
%         assert(false);
% end
% end


%%
classdef ConverterForBiLSTMLayer < nnet.internal.cnn.onnx.NNTLayerConverter
    % Class to convert a bilstmLayer into ONNX
    
    % Copyright 2018 The Mathworks, Inc.
    
    methods
        function this = ConverterForBiLSTMLayer(layerAnalyzer)
            this@nnet.internal.cnn.onnx.NNTLayerConverter(layerAnalyzer);
        end
        
        function [nodeProto, parameterInitializers, parameterInputs, networkInputs, networkOutputs, layerMap] = toOnnx(this, TensorNameMap)
            import nnet.internal.cnn.onnx.*
            
            [onnxName, nameChanged] = legalizeNNTName(this, this.NNTLayer.Name);
            NNTLayer = this.NNTLayer;
            hasSequenceOutput = isequal(NNTLayer.OutputMode, 'sequence');
            
            %% (1) Make the LSTM node
            nodeProto(1)            = NodeProto;
            nodeProto(1).op_type    = 'LSTM';
            nodeProto(1).name        = onnxName;
            input                   = {...
                this.InputLayerNames{1},... % X
                [onnxName '_W'],...
                [onnxName '_R'],...
                [onnxName '_B'],...
                '',...                      % sequence_Lens
                [onnxName '_initial_h'],...
                [onnxName '_initial_c']
                };
            nodeProto(1).input = mapTensorNames(this, input(:)', TensorNameMap);
            % Make the output name appear as either the sequence output or
            % the last output as appropriate.
            if hasSequenceOutput
                nodeProto(1).output = {onnxName};          % Output is Y=onnxName
            else
                nodeProto(1).output = {'', onnxName};      % Output is Y='', Y_h=onnxName
            end
            
            % Set attributes
            iofAct	= iTranslateActivationFunction(NNTLayer.GateActivationFunction);
            cellAct	= iTranslateActivationFunction(NNTLayer.StateActivationFunction);
            hidAct	= cellAct;
            nodeProto(1).attribute = [...
                makeAttributeProto('activations',       'STRINGS',	{iofAct, cellAct, hidAct, iofAct, cellAct, hidAct}),...
                makeAttributeProto('direction',         'STRING',    'bidirectional'),...
                makeAttributeProto('hidden_size',       'INT',       NNTLayer.NumHiddenUnits),...
                ];
            if this.OpsetVersion < 7
                nodeProto(1).attribute(end+1) = makeAttributeProto('output_sequence', 'INT', single(hasSequenceOutput));
            end
            
            % Get weight indices
            nH = NNTLayer.NumHiddenUnits;
            [cInd, iInd, fInd, oInd] = nnet.internal.cnn.util.gateIndices(nH);
            [forwardInd, backwardInd] = nnet.internal.cnn.util.forwardBackwardSequenceIndices(nH);
            iofcForward = forwardInd([iInd oInd fInd cInd]);
            iofcBackward = backwardInd([iInd oInd fInd cInd]);
            
            % Make parameterInitializers:
            % W
            inputWeights	= NNTLayer.InputWeights;
            W               = inputWeights(iofcForward,:);
            WB              = inputWeights(iofcBackward,:);
            WTensor         = zeros(2, 4*nH, NNTLayer.InputSize, 'single');
            WTensor(1,:,:)  = W;
            WTensor(2,:,:)  = WB;
            t1              = TensorProto;
            t1.name         = [onnxName '_W'];
            t1.data_type    = TensorProto_DataType.FLOAT;
            t1.raw_data     = rawData(single(WTensor));
            t1.dims         = dimVector(size(WTensor),3);
            
            % R
            recurrentWeights = NNTLayer.RecurrentWeights;
            W               = recurrentWeights(iofcForward,:);
            WB              = recurrentWeights(iofcBackward,:);
            RTensor         = zeros(2, 4*nH, nH, 'single');
            RTensor(1,:,:)  = W;
            RTensor(2,:,:)  = WB;
            t2              = TensorProto;
            t2.name         = [onnxName '_R'];
            t2.data_type    = TensorProto_DataType.FLOAT;
            t2.raw_data     = rawData(single(RTensor));
            t2.dims         = dimVector(size(RTensor),3);
            
            % B
            Bias            = NNTLayer.Bias;
            B               = Bias(iofcForward);
            RB          	= Bias(iofcBackward);
            nB              = numel(B);
            BTensor         = zeros(2, 8*nH, 'single');
            BTensor(1,1:nB)	= B;                        % Forward "recurrent biases" are left at 0.
            BTensor(2,1:nB)	= RB;                       % Backward "recurrent biases" are left at 0.
            t3              = TensorProto;
            t3.name         = [onnxName '_B'];
            t3.data_type    = TensorProto_DataType.FLOAT;
            t3.raw_data     = rawData(single(BTensor));
            t3.dims         = dimVector(size(BTensor),2);
            
            % initial_h
            HiddenState     = NNTLayer.HiddenState;                                         % NNT size is [2*nH, batchSize]
            batchSize       = size(HiddenState,2);
            initial_h       = permute(reshape(HiddenState, [2, nH, batchSize]), [1 3 2]);   % ONNX size is [2, batchSize, nH]
            t4              = TensorProto;
            t4.name         = [onnxName '_initial_h'];
            t4.data_type    = TensorProto_DataType.FLOAT;
            t4.raw_data     = rawData(single(initial_h));
            t4.dims         = dimVector(size(initial_h),3);
            
            % initial_c
            CellState       = NNTLayer.CellState;                                         % NNT size is [2*nH, batchSize]
            batchSize       = size(CellState,2);
            initial_c       = permute(reshape(CellState, [2, nH, batchSize]), [1 3 2]);   % ONNX size is [2, batchSize, nH]
            t5              = TensorProto;
            t5.name         = [onnxName '_initial_c'];
            t5.data_type    = TensorProto_DataType.FLOAT;
            t5.raw_data     = rawData(single(initial_c));
            t5.dims         = dimVector(size(initial_c),3);
            
            parameterInitializers = [t1 t2 t3 t4 t5];
            
            % Make parameterInputs
            parameterInputs = arrayfun(@makeValueInfoProtoFromTensorProto, parameterInitializers);
            
            networkInputs           = [];
            networkOutputs          = [];
            layerMap                = containers.Map;
            
            if nameChanged
                layerMap(this.NNTLayer.Name) = onnxName;
            end
        end
    end
end

function s = iTranslateActivationFunction(s)
switch s
    case 'sigmoid'
        s = 'Sigmoid';
    case 'softsign'
        s = 'Softsign';
    case 'tanh'
        s = 'Tanh';
    case 'hard-sigmoid'
        s = 'HardSigmoid';
    otherwise
        assert(false);
end
end
