function Y = vl_nnfscore(X,c,dzdy,varargin)
    opts = struct;
    opts.beta = 1;
    opts.instanceWeights = ones(size(c));
    opts.type = 'bin';
    opts.numClasses = 1;
    opts = vl_argparse(opts,varargin);

    if nargin <= 2 || isempty(dzdy)
        
        switch opts.type
            case 'bin'
                Xr = reshape(X,[],1,1,size(X,4));
                cr = reshape(c,[],1,1,size(c,4));
                P = (Xr>0); % Positives
                T = (cr>0);
                TP = sum(P & T); % true positives
                FP = sum(P & ~T); % false positives
                FN = sum(~P & T); % false negatives
                p = TP./(TP + FP);         % precision
                r = TP./(TP + FN);         % recall
                Y = (1+opts.beta^2).*(p.*r)./((opts.beta^2).*p + r);
                Y = nansum(Y,4);
            case 'class'
                if isa(X,'gpuArray')
                    Y = gpuArray.zeros([opts.numClasses,size(X,4)],classUnderlying(X));
                else
                    Y = zeros([opts.numClasses,size(X,4)],'like',X) ;
                end
                [~,chat] = max(X,[],3);
                Xr = reshape(chat,[],1,1,size(chat,4));
                cr = reshape(c,[],1,1,size(c,4));
                for i = 1:opts.numClasses
                    P = (Xr==i); % Positives
                    T = (cr==i);
                    TP = sum(P & T); % true positives
                    FP = sum(P & ~T); % false positives
                    FN = sum(~P & T); % false negatives
                    p = TP./(TP + FP);         % precision
                    r = TP./(TP + FN);         % recall
                    Yi = (1+opts.beta^2).*(p.*r)./((opts.beta^2).*p + r);
                    Y(i,:) = Yi;
                end
                Y = nanmean(Y,1);
                Y = nansum(Y,2);
                
            otherwise
                assert(false);
        end
        
    else
        Y = obj.zerosLike(X);
    end
end

function y = zerosLike(x)
    if isa(x,'gpuArray')
        y = gpuArray.zeros(size(x),classUnderlying(x)) ;
    else
        y = zeros(size(x),'like',x) ;
    end
end