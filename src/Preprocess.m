function J = Preprocess(I,windowSize,maxNorm)
%PREPROCESS Summary of this function goes here
%   Detailed explanation goes here
    switch nargin
        case 1
            error('Need at least two inputs.')
        case 2
            if isa(windowSize,'struct')
                maxNorm = windowSize.prep.maxNorm;
                windowSize = windowSize.prep.windowSize;
            end
        case 3
            % do nothing
        otherwise
            error('Incorrect number of inputs.')
    end
    if bitget(windowSize,1)~=1 || ~isscalar(windowSize)
        error('Input 2 (windowSize) must be an odd valued scalar.')
    end
    
    J = zeros(size(I),'like',I);
    for i = 1:size(J,3)
        K = I(:,:,i);
        J(:,:,i) = MaskedLocalResponse(K,true(size(K)),windowSize);
    end

    J(J>maxNorm) = maxNorm;
    J(J<-maxNorm) = -maxNorm;
end