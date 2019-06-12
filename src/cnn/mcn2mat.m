%%
function dag = mcn2mat(net,opts)

    dag = layerGraph;
    
    l = imageInputLayer([opts.prep.imgSize numel(opts.prep.channels)],...
                        'Normalization','none',...
                        'Name','input');
    dag = dag.addLayers(l);
    
    for i = 1:numel(net.layers)
        t = parseType(net.layers(i));
        switch t
            case 'dagnn.BatchNorm'
                g = strcmp({net.params.name},net.layers(i).params{1}); %index of scaling factor
                b = strcmp({net.params.name},net.layers(i).params{2}); %index of offset factor
                m = strcmp({net.params.name},net.layers(i).params{3}); %index of moments
                dag = batchnorm(dag,net.layers(i),...
                                net.params(g).value,...
                                net.params(b).value,...
                                net.params(m).value);
                            
            case 'dagnn.Concat'
                dag = concat(dag,net.layers(i));
                
            case 'dagnn.Conv'
                wind = strcmp({net.params.name},net.layers(i).params{1}); %index of weights
                bind = strcmp({net.params.name},net.layers(i).params{2}); %index of biases
                dag = conv(dag,net.layers(i),net.params(wind).value,net.params(bind).value);
                
            case 'dagnn.ConvTranspose'
                wind = strcmp({net.params.name},net.layers(i).params{1}); %index of weights
                dag = convt(dag,net.layers(i),net.params(wind).value);
                
            case 'dagnn.Crop'
                dag = crop(dag,net.layers(i),opts);
                
            case 'dagnn.DropOut'
                dag = dropout(dag,net.layers(i));
                
            case 'dagnn.Loss'
                % currently this function only translates networks for prediction.
                continue;
                
            case 'dagnn.Pooling'
                dag = pool(dag,net.layers(i));
                
            case 'dagnn.ReLU'
                dag = relu(dag,net.layers(i));
                
            case 'dagnn.SoftMax'
                % currently this function only translates networks for prediction.
                continue;
                
            case 'dagnn.Sum'
                dag = add(dag,net.layers(i));
                
            case 'dagnn.Fscore'
                % currently this function only translates networks for prediction.
                continue;
                
            case 'dagnn.L2Loss'
                % currently this function only translates networks for prediction.
                continue;
                
            otherwise
                error(['Unrecognized layer type on layer: ' num2str(i)]);
                
        end
    end
    
    % Add output layers
    inputs = {};
    for i = 1:height(dag.Connections)
        if any(strcmp(dag.Connections.Source,dag.Connections.Destination{i}))
            continue;
        end
        if contains(dag.Connections.Destination{i},'in') || contains(dag.Connections.Destination{i},'out')
            continue
        end
        inputs{end+1} = dag.Connections.Destination{i};
    end
    
    for i = 1:numel(inputs)
        l = copyOutput([inputs{i} '_out']);
        dag = dag.addLayers(l);
        dag = connectLayers(dag,inputs{i},[inputs{i} '_out/in']);
    end
    
    dag = assembleNetwork(dag);
end

%%
function t = parseType(layer)
    if isfield(layer,'type')
        t = layer.type;
    else
        t = class(layer.block);
    end
end

%%
function dag = conv(dag,layer,weights,bias)
    cfilt = layer.block.size;
    l = convolution2dLayer(cfilt(1:2),cfilt(4),...
                           'Stride',layer.block.stride,...
                           'DilationFactor',layer.block.dilate,...
                           'Padding',layer.block.pad,...
                           'NumChannels',cfilt(3),...
                           'Weights',weights,...
                           'Bias',reshape(bias,[1 1 cfilt(4)]),...
                           'Name',layer.outputs{1});
    dag = dag.addLayers(l);
    dag = connectLayers(dag,layer.inputs{1},[layer.outputs{1} '/in']);
end

%%
function dag = relu(dag,layer)
    if strcmpi(layer.block.method,'relu')
        if layer.block.leak>0
            l = leakyReluLayer(layer.block.leak,'Name',layer.outputs{1});
        else
            l = reluLayer('Name',layer.outputs{1});
        end
    elseif strcmpi(layer.block.method,'elu')
        l = eluLayer(layer.block.leak,'Name',layer.outputs{1});
    else
        error('Could not determine recognize relu type.')
    end
    dag = dag.addLayers(l);
    dag = connectLayers(dag,layer.inputs{1},[layer.outputs{1} '/in']);
end

%%
function dag = batchnorm(dag,layer,g,b,m)
    epsilon = layer.block.epsilon;
    l = batchNormalizationLayer('TrainedMean',reshape(m(:,1),1,1,size(m,1)),...
                                'TrainedVariance',reshape(m(:,2).^2,1,1,size(m,1))-epsilon,...
                                'Epsilon',epsilon,...
                                'Scale',reshape(g,1,1,numel(g)),...
                                'Offset',reshape(b,1,1,numel(b)),...
                                'Name',layer.outputs{1});
    dag = dag.addLayers(l);
    dag = connectLayers(dag,layer.inputs{1},[layer.outputs{1} '/in']);
end

%%
function dag = pool(dag,layer)
    switch layer.block.method
        case 'max'
            l = maxPooling2dLayer(layer.block.poolSize,...
                                  'Stride',layer.block.stride,...
                                  'Padding',layer.block.pad,...
                                  'Name',layer.outputs{1});
        case 'avg'
            l = averagePooling2dLayer(layer.block.poolSize,...
                                     'Stride',layer.block.stride,...
                                     'Padding',layer.block.pad,...
                                     'Name',layer.outputs{1});
        otherwise
            error('Did not recognize type of pooling layer.')
    end
    dag = dag.addLayers(l);
    dag = connectLayers(dag,layer.inputs{1},[layer.outputs{1} '/in']);
end

%%
function dag = add(dag,layer)
    l = additionLayer(numel(layer.inputs),'Name',layer.outputs{1});
    dag = dag.addLayers(l);
    for i = 1:numel(layer.inputs)
        dag = connectLayers(dag,layer.inputs{i},[layer.outputs{1} '/in' num2str(i)]);
    end
end

%%
function dag = convt(dag,layer,weights)
    cfilt = layer.block.size;
    if layer.block.numGroups>1
        out_size = layer.block.numGroups;
        if out_size ~= floor(out_size)
            error('Number of groups does not divide filters.')
        end
        w = zeros([cfilt(1:2) layer.block.numGroups cfilt(4)],'like',weights);
        stepsize = cfilt(4)/layer.block.numGroups;
        for i = 1:stepsize:cfilt(4)
            w(:,:,(i-1)/stepsize+1,i:i+stepsize-1) = weights(:,:,:,i:i+stepsize-1);
        end
        weights = w;
    else
        out_size = cfilt(4);
    end
    l = transposedConv2dLayer(cfilt(1:2),out_size,...
                              'Stride',layer.block.upsample,...
                              'Crop',layer.block.crop,...
                              'Weights',weights,...
                              'Bias',zeros(1,1,size(weights,3),'like',weights),...
                              'NumChannels',cfilt(4),...
                              'Name',layer.outputs{1});
    dag = dag.addLayers(l);
    dag = connectLayers(dag,layer.inputs{1},[layer.outputs{1} '/in']);
end

%%
function dag = concat(dag,layer)
    l = depthConcatenationLayer(numel(layer.inputs),'Name',layer.outputs{1});
    dag = dag.addLayers(l);
    for i = 1:numel(layer.inputs)
        dag = connectLayers(dag,layer.inputs{i},[layer.outputs{1} '/in' num2str(i)]);
    end
end

%%
function dag = dropout(dag,layer)
    l = dropoutLayer(layer.block.rate,'Name',layer.outputs{1});
    dag = dag.addLayers(l);
    dag = connectLayers(dag,layer.inputs{1},[layer.outputs{1} '/in']);
end

%%
function dag = crop(dag,layer,opts)
    l = constant2dCrop([(opts.prep.imgSize - opts.prep.lblSize)/2 opts.prep.lblSize],layer.outputs{1});
    dag = dag.addLayers(l);
    dag = connectLayers(dag,layer.inputs{1},[layer.outputs{1} '/in']);
end