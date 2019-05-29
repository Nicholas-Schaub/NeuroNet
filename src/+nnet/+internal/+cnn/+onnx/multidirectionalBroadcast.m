function dim = multidirectionalBroadcast(dim)
%   Copyright 2018-2019 The MathWorks, Inc.
switch numel(dim)
    case 1
        % Assume it's the C dimension only. Make it 1C11:
        dim = [1 dim 1 1];
    case 2
        if dim(1)==1
            % Assume it's flattened: 1C. Make it NCHW = 1C11.
            dim = [dim 1 1];
        else
            % Assume it's CK. Make it 1CK1. (Not sure we have a use case
            % for this.)
            dim = [1 dim 1];
        end
    case 3
        % Assume it's CHW. Make it 1CHW
        dim = [1 dim];
    case 4
        % It's already 4D. Do nothing.
end
end