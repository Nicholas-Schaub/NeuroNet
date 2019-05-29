function v = dimVector(sz, numDims)
% Takes a MATLAB size vector, extends it to numDims and converts it to
% int64.
assert(numel(sz) <= numDims);
v = int64([sz ones(1,numDims-numel(sz))]);
end
