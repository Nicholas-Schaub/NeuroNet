function [Ix,Jx] = Tile(I,J,sizes)
%TILE Summary of this function goes here
%   Detailed explanation goes here

    switch nargin
        case 1
            error('Need at least two inputs.')
        case 2
            processLabel = false;
            if isa(J,'struct')
                im_size = J.prep.imgSize;
                lbl_size = J.prep.lblSize;
            elseif numel(J)~=4
                error('Expected input 2 to be a four element vector indicating height and width of tiling.');
            else
                im_size = J(1:2);
                lbl_size = J(3:4);
            end
        case 3
            processLabel = true;
            if (size(I,1) ~= size(J,1)) || (size(I,2) ~= size(J,2))
                error('Input images are not the same size.');
            end
            if isa(sizes,'struct')
                im_size = sizes.prep.imgSize;
                lbl_size = sizes.prep.lblSize;
            elseif numel(J)~=4
                error('Expected input 2 to be a four element vector indicating height and width of tiling.');
            else
                im_size = sizes(1:2);
                lbl_size = sizes(3:4);
            end
        otherwise
            error('Incorrect number of inputs.')
    end
    
    % Pad the image for tiling
    xypad = (im_size-lbl_size)/2;
    xypad(1) = xypad(1) + (lbl_size(1) - mod(size(I,1),lbl_size(1)))/2;
    xypad(2) = xypad(2) + (lbl_size(2) - mod(size(I,2),lbl_size(2)))/2;
    pad_img = padarray(I,floor(xypad),'pre','symmetric');
    pad_img = padarray(pad_img,ceil(xypad),'post','symmetric');
    
    if processLabel
        xypad(1) = (lbl_size(1) - mod(size(I,1),lbl_size(1)))/2;
        xypad(2) = (lbl_size(2) - mod(size(I,2),lbl_size(2)))/2;
        pad_lbl = padarray(J,floor(xypad),'pre','symmetric');
        pad_lbl = padarray(pad_lbl,ceil(xypad),'post','symmetric');
    end
    
    % Generate the indices for the image tile
    [nRows, nCols, ~] = size(pad_img);
    [cindex,rindex] = meshgrid(0:im_size(2)-1,0:im_size(1)-1);
    img_ind = cindex(:)*nRows+rindex(:);
    [cindex,rindex] = meshgrid(0:lbl_size(2):nCols-lbl_size(2),...
                               1:lbl_size(1):nRows-lbl_size(1)+1); %for image data
    raw_indices = nRows*cindex(:)+rindex(:);
    img_indices = bsxfun(@plus,img_ind(:),raw_indices(:)');
    
    if processLabel
        [segRows, segCols, ~] = size(pad_lbl);
        [cindex,rindex] = meshgrid(0:lbl_size(2)-1,0:lbl_size(1)-1);
        seg_ind = cindex(:)*segRows+rindex(:);
        [cindex,rindex] = meshgrid(0:lbl_size(2):segCols-1,...
                                   1:lbl_size(1):segRows); %for image data
        raw_indices = segRows*cindex(:)+rindex(:);
        seg_indices = bsxfun(@plus,seg_ind(:),raw_indices(:)');
    end

    % Generate the output
    Ix = zeros(im_size(1),im_size(2),size(I,3),size(img_indices,2),'like',I);
    for i = 1:size(I,3)
        Ix(:,:,i,:) = reshape(pad_img(img_indices+nRows*nCols*(i-1)),im_size(1),im_size(2),1,size(img_indices,2));
    end
    
    if processLabel
        Jx = zeros(lbl_size(1),lbl_size(2),size(J,3),size(seg_indices,2),'like',J);
        for i = 1:size(J,3)
            Jx(:,:,i,:) = reshape(pad_lbl(seg_indices+segRows*segCols*(i-1)),lbl_size(1),lbl_size(2),1,size(seg_indices,2));
        end
    end
    
end