function uint8vector = rawData(A)
% Reorders a MATLAB numeric multidimensional array into row-major order,
% and then typecasts it into a vector of uint8's.
assert(isnumeric(A));
rowMajorData = permute(A, fliplr(1:ndims(A)));
uint8vector = typecast(rowMajorData(:)', 'uint8');
end
