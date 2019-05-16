function [net,outVar] = createRRLayer(net, inVar, outVar, inChan, outChan, filtSize, kappa, reluType, leak)
%CREATERRLAYER Create reduced rank convolutional layer

    if mod(filtSize,2)==0
        error('Size must be odd')
    end

    row_layer = [outVar '_row'];
    row_relu = [row_layer '_relu'];
    col_layer = [outVar '_col'];
    col_relu = [col_layer '_relu'];
    
    r = dagnn.Conv('size',[filtSize 1 inChan kappa],...
                   'pad',0,'stride',1,'hasBias',true);
    c = dagnn.Conv('size',[1 filtSize kappa outChan],...
                   'pad',floor(filtSize/2),'stride',1,'hasBias',true);
                 
    net.addLayer(row_layer,r,...
                 {inVar},{row_layer},...
                 {[row_layer '_f'] [row_layer '_b']});
    net = init_params(net);
    net.addLayer(row_relu,dagnn.ReLU('leak',leak,'method',reluType),...
                 {row_layer},{row_relu});
    net = init_params(net);
    net.addLayer(col_layer,c,...
                 {row_relu},{col_layer},...
                 {[col_layer '_f'] [col_layer '_b']});
    net = init_params(net);
    net.addLayer(col_relu,dagnn.ReLU('leak',leak,'method',reluType),...
                 {col_layer},{col_relu});
    net = init_params(net);
    
    outVar = col_relu;
    
end