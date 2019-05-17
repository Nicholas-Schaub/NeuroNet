function Y = vl_nnL2(X,c,dzdy,varargin)

opts = struct;
opts.loss = 'L2';
opts.instanceWeights = ones(size(c));
opts = vl_argparse(opts,varargin);

if nargin <= 2 || isempty(dzdy)
    switch opts.loss
        case 'L2'
            Y = 0.5*sum(opts.instanceWeights.*(X-c).^2);
        case 'R2'
            Y = 1 - sum((X-c).^2)/sum((c-mean(c)).^2);
            if Y<0
                Y = 0;
            end
        case 'RMSE'
            Y = sqrt(mean((X-c).^2));
    end
else
    switch opts.loss
        case 'L2'
            Y = +(X-c)*dzdy.*opts.instanceWeights;
            Y = reshape(Y,size(X));
        case 'R2'
            Y = zerosLike(X);
        case 'RMSE'
            Y = zerosLike(X);
    end
end

% --------------------------------------------------------------------
function y = zerosLike(x)
% --------------------------------------------------------------------
if isa(x,'gpuArray')
  y = gpuArray.zeros(size(x),classUnderlying(x)) ;
else
  y = zeros(size(x),'like',x) ;
end