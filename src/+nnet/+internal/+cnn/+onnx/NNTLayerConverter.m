classdef NNTLayerConverter
    % Class to convert a NNT Layer to a nnet.internal.cnn.onnx.NodeProto
    
    % Copyright 2018 The Mathworks, Inc.
    
    %% API
    properties(Dependent)
        NNTLayer            % The NNT layer being converted.
        InputLayerNames     % (cellstr) The names of the layers (really tensors) sending input connections to this layer.
        InputLayerSizes     % (cell array) Sizes of the layers (really tensors) sending input connections to this layer.
        OutputSize          % Size of the output tensor of this layer.
    end
    
    properties
        OpsetVersion        % Integer
        IsRecurrentNetwork
    end
    
    %% Override 'toOnnx()' to convert an NNT layer to ONNX.
    methods
        % In toOnnx(),
        %   * nodeProto is an array of nnet.internal.cnn.onnx.NodeProto.
        %   * parameterInitializers is an array of  nnet.internal.cnn.onnx.TensorProto.
        %   * parameterInputs is an array of nnet.internal.cnn.onnx.NodeProto.
        %   * networkInputs is an array of nnet.internal.cnn.onnx.ValueInfoProto,
        %   which should be present only when this layer defines input to the network as a whole.
        %   * networkOutputs is an array of nnet.internal.cnn.onnx.ValueInfoProto,
        %   which should be present only when this layer defines output of the network as a whole.
        %   * TensorNameMap is a containers.Map that translates between NNT
        %   output names and ONNX tensor names.
        %   * layerTensorNameMap is a containers.Map with additions to be
        %   added for this layer.
        function [nodeProto, parameterInitializers, parameterInputs,...
                networkInputs, networkOutputs, layerTensorNameMap] = toOnnx(this, TensorNameMap)
            nodeProto               = [];
            parameterInitializers   = [];
            parameterInputs         = [];
            networkInputs           = [];
            networkOutputs          = [];
            layerTensorNameMap      = containers.Map;
        end
    end
    
    methods(Static)
        function converter = makeLayerConverter(layerAnalyzer, OpsetVersion, IsRecurrentNetwork)
            % Factory method to create a converter for a specific layer.
            import nnet.internal.cnn.onnx.*
            assert(isa(layerAnalyzer, 'nnet.internal.cnn.analyzer.util.LayerAnalyzer'));
            nntLayer = layerAnalyzer.ExternalLayer;
            assert(isa(nntLayer, 'nnet.cnn.layer.Layer'));
            
            % DLT Layers
            if isa(nntLayer, 'nnet.cnn.layer.AdditionLayer')
                converterConstructor = @ConverterForAdditionLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.AveragePooling2DLayer')
                converterConstructor = @ConverterForAveragePooling2DLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.BatchNormalizationLayer')
                converterConstructor = @ConverterForBatchNormalizationLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.BiLSTMLayer')
                converterConstructor = @ConverterForBiLSTMLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.ClassificationOutputLayer')
                converterConstructor = @ConverterForClassificationOutputLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.ClippedReLULayer')
                converterConstructor = @ConverterForClippedReLULayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.Convolution2DLayer')
                converterConstructor = @ConverterForConvolution2DLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.CrossChannelNormalizationLayer')
                converterConstructor = @ConverterForCrossChannelNormalizationLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.DepthConcatenationLayer')
                converterConstructor = @ConverterForDepthConcatenationLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.DropoutLayer')
                converterConstructor = @ConverterForDropoutLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.FullyConnectedLayer')
                converterConstructor = @ConverterForFullyConnectedLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.GroupedConvolution2DLayer')
                converterConstructor = @ConverterForGroupedConvolution2DLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.ImageInputLayer')
                converterConstructor = @ConverterForImageInputLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.LeakyReLULayer')
                converterConstructor = @ConverterForLeakyReLULayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.LSTMLayer')
                converterConstructor = @ConverterForLSTMLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.MaxPooling2DLayer')
                converterConstructor = @ConverterForMaxPooling2DLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.MaxUnpooling2DLayer')
                converterConstructor = @ConverterForMaxUnpooling2DLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.RegressionOutputLayer')
                converterConstructor = @ConverterForRegressionOutputLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.ReLULayer')
                converterConstructor = @ConverterForReLULayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.SequenceInputLayer')
                converterConstructor = @ConverterForSequenceInputLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.SigmoidLayer')
                converterConstructor = @ConverterForSigmoidLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.SoftmaxLayer')
                converterConstructor = @ConverterForSoftmaxLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.TransposedConvolution2DLayer')
                converterConstructor = @ConverterForTransposedConvolution2DLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.ELULayer')
                converterConstructor = @ConverterForELULayer;
            elseif isa(nntLayer, 'constant2dCrop')
                converterConstructor = @ConverterForConstant2DCrop;
                
                % CVST Layers
            elseif isa(nntLayer, 'nnet.cnn.layer.Crop2DLayer')
                converterConstructor = @ConverterForCrop2DLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.PixelClassificationLayer')
                converterConstructor = @ConverterForPixelClassificationLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.RCNNBoxRegressionLayer')
                converterConstructor = @ConverterForRCNNBoxRegressionLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.ROIInputLayer')
                converterConstructor = @ConverterForROIInputLayer;
            elseif isa(nntLayer, 'nnet.cnn.layer.ROIMaxPooling2DLayer')
                converterConstructor = @ConverterForROIMaxPooling2DLayer;
                
                % Keras importer layers
            elseif isa(nntLayer, 'nnet.keras.layer.FlattenCStyleLayer')
                converterConstructor = @ConverterForKerasFlattenCStyleLayer;
            elseif isa(nntLayer, 'nnet.keras.layer.GlobalAveragePooling2dLayer')
                converterConstructor = @ConverterForKerasGlobalAveragePooling2dLayer;
            elseif isa(nntLayer, 'nnet.keras.layer.SigmoidLayer')
                converterConstructor = @ConverterForKerasSigmoidLayer;
            elseif isa(nntLayer, 'nnet.keras.layer.TanhLayer')
                converterConstructor = @ConverterForKerasTanhLayer;
            elseif isa(nntLayer, 'nnet.keras.layer.ZeroPadding2dLayer')
                converterConstructor = @ConverterForKerasZeroPadding2dLayer;
                
                % ONNX converter layers
            elseif isa(nntLayer, 'nnet.onnx.layer.BiasLayer')
                converterConstructor = @ConverterForONNXBiasLayer;
            elseif isa(nntLayer, 'nnet.onnx.layer.ClipLayer')
                converterConstructor = @ConverterForONNXClipLayer;
            elseif isa(nntLayer, 'nnet.onnx.layer.FlattenLayer')
                converterConstructor = @ConverterForONNXFlattenLayer;
            elseif isa(nntLayer, 'nnet.onnx.layer.GlobalAveragePooling2dLayer')
                converterConstructor = @ConverterForONNXGlobalAveragePooling2dLayer;
            elseif isa(nntLayer, 'nnet.onnx.layer.IdentityLayer')
                converterConstructor = @ConverterForONNXIdentityLayer;
            elseif isa(nntLayer, 'nnet.onnx.layer.ImageScalerLayer')
                converterConstructor = @ConverterForONNXImageScalerLayer;
            elseif isa(nntLayer, 'nnet.onnx.layer.ElementwiseAffineLayer')
                converterConstructor = @ConverterForONNXElementwiseAffineLayer;
            elseif isa(nntLayer, 'nnet.onnx.layer.PreluLayer')
                converterConstructor = @ConverterForONNXPreluLayer;
            elseif isa(nntLayer, 'nnet.onnx.layer.ScalingFactorLayer')
                converterConstructor = @ConverterForONNXScalingFactorLayer;
            elseif isa(nntLayer, 'nnet.onnx.layer.SigmoidLayer')
                converterConstructor = @ConverterForONNXSigmoidLayer;
            elseif isa(nntLayer, 'nnet.onnx.layer.TanhLayer')
                converterConstructor = @ConverterForONNXTanhLayer;
                
                % Densenet201 layers
            elseif isa(nntLayer, 'nnet.densenet201.layer.DenseNet201PreprocessingLayer')
                converterConstructor = @ConverterForDenseNet201PreprocessingLayer;
                
                % Inceptionresnetv2 layers
            elseif isa(nntLayer, 'nnet.inceptionresnetv2.layer.ScalingLayer')
                converterConstructor = @ConverterForInceptionResNetV2ScalingLayer;
            elseif isa(nntLayer, 'nnet.inceptionresnetv2.layer.ScalingFactorLayer')
                converterConstructor = @ConverterForInceptionResNetV2ScalingFactorLayer;
                
                % Inceptionv3 layers
            elseif isa(nntLayer, 'nnet.inceptionv3.layer.ScalingLayer')
                converterConstructor = @ConverterForInceptionV3ScalingLayer;
                
                % mobilenetv2 layers
            elseif isa(nntLayer, 'nnet.mobilenetv2.layer.MobileNetV2PreprocessingLayer')
                converterConstructor = @ConverterForMobileNetV2PreprocessingLayer;
                
                % nasnetmobile layers
            elseif isa(nntLayer, 'nnet.nasnetmobile.layer.NASNetMobileCrop2dLayer')
                converterConstructor = @ConverterForNASNetMobileCrop2dLayer;
            elseif isa(nntLayer, 'nnet.nasnetmobile.layer.NASNetMobilePreprocessingLayer')
                converterConstructor = @ConverterForNASNetMobilePreprocessingLayer;
            elseif isa(nntLayer, 'nnet.nasnetmobile.layer.NASNetMobileZeroPadding2dLayer')
                converterConstructor = @ConverterForNASNetMobileZeroPadding2dLayer;
                
                % Resnet18 layers
            elseif isa(nntLayer, 'nnet.resnet18.layer.ResNet18PreprocessingLayer')
                converterConstructor = @ConverterForResNet18PreprocessingLayer;
                
                % xception layers
            elseif isa(nntLayer, 'nnet.xception.layer.XceptionPreprocessingLayer')
                converterConstructor = @ConverterForXceptionPreprocessingLayer;
                
                % Unknown (user-defined) custom layers
                %             elseif layerAnalyzer.IsCustomLayer
                %                 converterConstructor = @ConverterForCustomLayer;
                %                 warning(message('nnet_cnn_onnx:onnx:ExportingCustomLayer', class(nntLayer)));
                % Unsupported layer:
            else
                converterConstructor = @ConverterForUnsupportedLayer;
                warning(message('nnet_cnn_onnx:onnx:ExportingPlaceholderLayer', class(nntLayer)));
            end
            converter = converterConstructor(layerAnalyzer);
            converter.OpsetVersion = OpsetVersion;
            converter.IsRecurrentNetwork = IsRecurrentNetwork;
        end
    end
    
    %% INTERNALS
    properties(Access=private)
        LayerAnalyzer       % A LayerAnalyzer for the NNT layer being converted.
    end
    
    methods
        function this = NNTLayerConverter(layerAnalyzer)
            this.LayerAnalyzer = layerAnalyzer;
        end
        
        function [name, changed] = legalizeNNTName(~, nntName)
            % Make the NNT name a legal ONNX (C) name. Replace all chars
            % that are not letters or digits with underscores. If it starts
            % with a digit, prepend an underscore.
            name = nntName;
            toReplace = ~(isletter(name) | isdigit(name));
            name(toReplace) = '_';
            if isdigit(name(1))
                name = ['_' name];
            end
            changed = ~isequal(name, nntName);
        end
        
        function namesCell = mapTensorNames(~, namesCell, TensorNameMap)
            for i = 1:numel(namesCell)
                if isKey(TensorNameMap, namesCell{i})
                    namesCell{i} = TensorNameMap(namesCell{i});
                end
            end
        end
        
        function layer = get.NNTLayer(this)
            layer = this.LayerAnalyzer.ExternalLayer;
        end
        
        function inputNames = get.InputLayerNames(this)
            inputNames = this.LayerAnalyzer.Inputs.Source;
            inputNames = cellfun(@char, inputNames, 'UniformOutput', false);
        end
        
        function inputSizes = get.InputLayerSizes(this)
            inputSizes = this.LayerAnalyzer.Inputs.Size;
        end
        
        function outputSize = get.OutputSize(this)
            outputSize = this.LayerAnalyzer.Outputs.Size;
        end
    end
end

function flags = isdigit(c)
% Return true for every char in c that is a digit
flags = (c>='0' & c<='9');
end
