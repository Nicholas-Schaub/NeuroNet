function [net] = init_params(net)
    p = net.getParamIndex(net.layers(end).params) ;
    params = net.layers(end).block.initParams() ;
    switch net.device
        case 'cpu'
            params = cellfun(@gather, params, 'UniformOutput', false) ;
        case 'gpu'
            params = cellfun(@gpuArray, params, 'UniformOutput', false) ;
    end
    disp(['Initializing: ' net.layers(end).name]);
    [net.params(p).value] = deal(params{:}) ;
end