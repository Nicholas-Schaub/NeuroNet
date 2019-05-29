function NNTLayer = translateLRN(node,LayerName, OpsetVersion)
    
      %Get the attributes
    
       attributeNames = arrayfun(@(a) a.name, node.attribute,'UniformOutput',false);
       attributeInt = arrayfun(@(a) a.i, node.attribute,'UniformOutput',false);
       attributeFloat = arrayfun(@(a) a.f, node.attribute,'UniformOutput',false);
       attributeIntMap = containers.Map(attributeNames, attributeInt);
       attributeFloatMap = containers.Map(attributeNames, attributeFloat);
      
       if ismember('bias', attributeNames)
           K =attributeFloatMap('bias');
       else
           K = 1.0;
       end
       
       beta = attributeFloatMap('beta');
       alpha = attributeFloatMap('alpha');
       windowSize =  attributeIntMap('size'); 
       NNTLayer =  crossChannelNormalizationLayer(windowSize, 'Alpha', alpha, 'Beta', beta, 'K',K, 'Name', LayerName);
       
end