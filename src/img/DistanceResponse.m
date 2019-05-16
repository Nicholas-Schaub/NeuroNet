function response = DistanceResponse(mask,thresh)
    response = bwdist(mask==0);
    response(isinf(response)) = double(thresh);
    response(mask) = thresh.*double(response(mask)>thresh) + ...
                     response(mask).*double(response(mask)<=thresh);
end