function [norm,labels,weights] = Weight(I,S,opts)
%LRWeight Weight pixels according to pixel intensity
    
    if nargin==2
        opts = appParams();
    end
    
    I = single(I(:,:,opts.prep.channels));
    S = single(S);
    
    if opts.prep.windowSize/2==round(opts.prep.windowSize/2)
        opts.prep.windowSize = opts.prep.windowSize-1;
    end
    
    % Initialize output variables
    labels = ones(size(S));
    weights = ones(size(S));
    
    % Threshold the label image if needed
    if ~isempty(opts.prep.binThreshold)
        if size(opts.prep.binThreshold,2)==1 % binary thresholding
            for i = 1:size(opts.prep.binThreshold,1)
                labels(S<=opts.prep.binThreshold) = opts.prep.backVal;
                labels(S>opts.prep.binThreshold) = opts.prep.foreVal;
            end
        else % generate classification image
            for i = 1:size(opts.prep.binThreshold,1)
                [N,~,labels] = histcounts(S,opts.prep.binThreshold);
            end
        end
    else
        labels = S;
    end
    
    % Add weights for network that assigns attributes to a pixel.
    % Currently only supports a single attribute (binary segmentation)
    % TODO: Add support for multiple attributes, classes, and regression
    if mod(opts.network.type,3)==0
        fore_mask = labels==opts.prep.foreVal;
        back_mask = labels==opts.prep.backVal;
        switch opts.prep.foreWeightType
            case 'intensity'
                r = opts.prep.foreWeightThreshDir.*MaskedLocalResponse(I,fore_mask,opts.prep.windowSize);
                if ~isempty(opts.prep.foreWeightThresh)
                    r(r<(opts.prep.foreWeightThreshDir*opts.prep.foreWeightThresh)) = opts.prep.foreWeightThresh;
                end
                r = abs(r);
                weights(fore_mask) = r(fore_mask).^opts.prep.foreWeightAmp + 1;
            case 'distance'
                r = DistanceResponse(fore_mask,opts.prep.foreWeightThresh);
                weights(fore_mask) = r(fore_mask).^opts.prep.foreWeightAmp;
        end
        if (opts.prep.foreWeightAmp<0)
            weights(fore_mask) = weights(fore_mask).*(opts.prep.maxWeight);
        end
        switch opts.prep.backWeightType
            case 'intensity'
                r = opts.prep.backWeightThreshDir.*MaskedLocalResponse(I,back_mask,opts.prep.windowSize);
                if ~isempty(opts.prep.backWeightThresh)
                    r(r<(opts.prep.backWeightThreshDir*opts.prep.backWeightThresh)) = opts.prep.backWeightThresh;
                end
                r = abs(r);
                weights(back_mask) = r(back_mask).^opts.prep.backWeightAmp + 1;
            case 'distance'
                r = DistanceResponse(back_mask,opts.prep.backWeightThresh);
                weights(back_mask) = r(back_mask).^opts.prep.backWeightAmp;
        end
        if (opts.prep.backWeightAmp<0)
            weights(back_mask) = weights(back_mask).*(opts.prep.maxWeight);
        end
        
        weights(weights>opts.prep.maxWeight) = opts.prep.maxWeight;
    
        if opts.prep.normWeights
            normVal = sum(weights(fore_mask))/sum(weights(back_mask));
            weights(back_mask) = weights(back_mask).*normVal;
        end

        if any(isnan(weights(:)))
            error('Some weights are NaN.')
        end

        if opts.prep.normImage
            norm = MaskedLocalResponse(I,true(size(I)),opts.prep.windowSize);
            norm(norm>opts.prep.maxNorm) = opts.prep.maxNorm;
            norm(norm<-opts.prep.maxNorm) = -opts.prep.maxNorm;
        else
            norm = I;
        end
    end
    
    if mod(opts.network.type,3)==1
        
        if opts.prep.normImage
            norm = MaskedLocalResponse(I,I>0,opts.prep.windowSize);
            norm(norm>opts.prep.maxNorm) = opts.prep.maxNorm;
            norm(norm<-opts.prep.maxNorm) = -opts.prep.maxNorm;
        else
            norm = I;
        end
        
        weight_count = zeros(length(N),1);
        for i = 1:length(N)
            r = DistanceResponse(labels==i,opts.prep.foreWeightThresh);
            weights(labels==i) = r(labels==i).^opts.prep.foreWeightAmp;
            weight_count(i) = sum(double(weights(labels==i)));
        end
        
        if opts.prep.normWeights
            for i = 1:length(N)
                weights(labels==i) = weights(labels==i).*(sum(weight_count)/weight_count(i));
            end
        end
    end
    
end