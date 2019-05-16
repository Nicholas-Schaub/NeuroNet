function y = get_batch_diagnose(imdb,batch,opts)
%GET_BATCH Summary of this function goes here
%   Detailed explanation goes here

    if opts.prep.preLoad
        im = imdb.images.data(:,:,:,batch);
        cadasilLabels = imdb.images.cadasilLabel(:,:,:,batch);
        lobeLabels = imdb.images.regionLabel(:,:,:,batch);
        mutationLabels = imdb.images.mutationLabel(:,:,:,batch);
        cadasilWeights = imdb.images.cadasilWeights(:,:,:,batch);
        lobeWeights = imdb.images.regionWeights(:,:,:,batch);
        mutationWeights = imdb.images.mutationWeights(:,:,:,batch);
    else
        error('Only preloaded images are accepted for this code.')
    end
    
    if opts.useGpu
        im = gpuArray(im);
        cadasilLabels = gpuArray(cadasilLabels);
        lobeLabels = gpuArray(lobeLabels);
        mutationLabels = gpuArray(mutationLabels);
        cadasilWeights = gpuArray(cadasilWeights);
        lobeWeights = gpuArray(lobeWeights);
        mutationWeights = gpuArray(mutationWeights);
    end
    
    if imdb.images.set(batch(1))==1
        
        if opts.jitter.flipHorizontal && rand > 0.5
            im = fliplr(im);
        end
        
        if opts.jitter.flipVertical && rand > 0.5
            im = flipud(im);
        end
        
    end
    
    y = {'input', im,...
         'cadasil_label', cadasilLabels,'cadasil_weight', cadasilWeights,...
         'lobe_label',lobeLabels,'lobe_weight', lobeWeights,...
         'mutation_label',mutationLabels,'mutation_weight', mutationWeights} ;
end