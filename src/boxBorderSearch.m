function [ perimeter ] = boxBorderSearch(lblImage,boxSize)
    % Get image size and box size values
	[nRows, nCols] = size(lblImage);
    fOffset = floor(boxSize/2);
    cOffset = ceil(boxSize/2);
    
    % Create the integral image
    intI = zeros(nRows+1,nCols+1);
    intI(2:end,2:end) = cumsum(cumsum(double(lblImage),1),2);
    
    % Create indices for the original image (ii,jj)
    [cols,rows] = meshgrid(0:nCols-boxSize,1:nRows-boxSize+1);
    ul = (nRows+1)*cols(:)+rows(:);
    ur = ul+(boxSize)*(nRows+1);
    ll = ul+boxSize;
    lr = ur+boxSize;
    
    % Get the sum of local neighborhood defined by boxSize
    vals = (intI(ul)+ intI(lr) - intI(ur) - intI(ll));
    
    % Divide the pixel averages by the pixel value. If a pixel is not
    % bordering another object, then this value will be equal to
    % boxSize*boxSize.
    pixMask = reshape(vals,[nRows-2*fOffset nCols-2*fOffset])./double(lblImage(cOffset:end-fOffset,cOffset:end-fOffset));
    pixMask = padarray(pixMask,[fOffset fOffset]);
    thresh = boxSize*boxSize;
    perimeter_indices = find(pixMask~=thresh);
    perimeter = zeros(size(lblImage));
    perimeter(perimeter_indices) = lblImage(perimeter_indices);
end