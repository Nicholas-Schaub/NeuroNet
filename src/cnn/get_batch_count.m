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
    counts = imdb.images.count(:,:,:,batch);
    count_weights = ones(size(counts));

    if opts.useGpu
        im = gpuArray(im);
        counts = gpuArray(counts);
        count_weights = gpuArray(count_weights);
    end
    
    if imdb.images.set(batch(1))==1
        
        if opts.jitter.flipHorizontal && rand > 0.5
            im = fliplr(im);
        end
        
        if opts.jitter.flipVertical && rand > 0.5
            im = flipud(im);
        end
        
    end
    
    y = {'input', im, 'count', counts, 'count_weight', count_weights} ;
    
    
end