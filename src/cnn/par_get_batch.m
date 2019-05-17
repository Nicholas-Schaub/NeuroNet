function y = par_get_batch(imdb,batch,opts)    
    persistent P
    
    if nargout==0
        if ~opts.loadParallel
            error('Network training is trying to prefetch images but parallel loading setting is false.');
        else
            if isempty(batch)
                P = [];
            else
                p = gcp();
                P = parfeval(p,opts.prep.getBatch,1,imdb,batch);
            end
            return;
        end
    else
        if ~opts.loadParallel || isempty(P)
            y = opts.prep.getBatch(imdb,batch);
        else
            y = fetchOutputs(P);
        end
    end
end