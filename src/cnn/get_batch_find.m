function y = get_batch(imdb,batch,opts)
%GET_BATCH Summary of this function goes here
%   Detailed explanation goes here

%     if opts.prep.preLoad
%         im = imdb.images.data(:,:,:,batch);
%         labels = imdb.images.label(:,:,:,batch);
%         weights = imdb.images.weight(:,:,:,batch);
%     else
%         error('Only preloaded images are accepted for this code.')
%     end
    
    im = imdb.images.data(:,:,:,batch);
    labels = imdb.images.label(:,:,:,batch);
    weights = imdb.images.weight(:,:,:,batch);

    if opts.useGpu
        im = gpuArray(im);
        labels = gpuArray(labels);
        weights = gpuArray(weights);
    end
    
    if imdb.images.set(batch(1))==1
        
        if opts.jitter.flipHorizontal && rand > 0.5
            im = fliplr(im);
            labels = fliplr(labels);
            weights = fliplr(weights);
        end
        
        if opts.jitter.flipVertical && rand > 0.5
            im = flipud(im);
            labels = flipud(labels);
            weights = flipud(weights);
        end
        
    end
    
    if opts.prep.mixWeightsWithLabels
        labels = labels.*weights;
        weights = ones(size(weights),'single');
    end
    y = {'input', im, 'label', labels, 'weight', weights} ;
    
    
end