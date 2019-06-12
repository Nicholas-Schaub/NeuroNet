function [Ix,Jx] = Tile3(I,J,sizes)
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
            elseif numel(J)~=6
                error('Expected input 2 to be a four element vector indicating height and width of tiling.');
            else
                im_size = J(1:3);
                lbl_size = J(4:6);
            end
        case 3
            processLabel = true;
            if (size(I,1) ~= size(J,1)) || (size(I,2) ~= size(J,2)) || (size(I,3) ~= size(J,3))
                error('Input images are not the same size.');
            end
            if isa(sizes,'struct')
                im_size = sizes.prep.imgSize;
                lbl_size = sizes.prep.lblSize;
            elseif numel(sizes)~=6
                error('Expected input 2 to be a six element vector indicating height, width, and depth of tiling.');
            else
                im_size = sizes(1:3);
                lbl_size = sizes(4:6);
            end
        otherwise
            error('Incorrect number of inputs.')
    end
    
    % Get size information
    I_size = size(I);
    if length(I_size)==3
        I_size(end+1) = 1;
    end
    I_dtype = class(I);
    
    % Pad the image for tiling
    xypad = (im_size-lbl_size)/2;
    xypad(1) = xypad(1) + (lbl_size(1) - mod(I_size(1),lbl_size(1)))/2;
    xypad(2) = xypad(2) + (lbl_size(2) - mod(I_size(2),lbl_size(2)))/2;
    xypad(3) = xypad(3) + (lbl_size(3) - mod(I_size(3),lbl_size(3)))/2;
    pad_img = padarray(I,floor(xypad(1:3)),'pre','symmetric');
    pad_img = padarray(pad_img,ceil(xypad(1:3)),'post','symmetric');
    clear I;
    
    if processLabel
        J_size = size(J);
        if length(J_size)==3
            J_size(end+1) = 1;
        end
        J_dtype = class(J);
        xypad(1) = (lbl_size(1) - mod(I_size(1),lbl_size(1)))/2;
        xypad(2) = (lbl_size(2) - mod(I_size(2),lbl_size(2)))/2;
        xypad(3) = (lbl_size(3) - mod(I_size(3),lbl_size(3)))/2;
        pad_lbl = padarray(J,floor(xypad),'pre','symmetric');
        pad_lbl = padarray(pad_lbl,ceil(xypad),'post','symmetric');
        clear J;
    end
    
    % Generate the indices for the image tile
    [nRows, nCols, nDepth, ~] = size(pad_img);
    [cindex,rindex, dindex] = meshgrid(0:im_size(2)-1,0:im_size(1)-1,0:im_size(3)-1);
    if prod(I_size) < 2^32-1
        cindex = uint32(cindex);
        rindex = uint32(rindex);
        dindex = uint32(dindex);
    end
    img_ind = dindex(:)*nCols*nRows+cindex(:)*nRows+rindex(:);
    [cindex,rindex,dindex] = meshgrid(0:lbl_size(2):nCols-lbl_size(2)-1,...
                                      1:lbl_size(1):nRows-lbl_size(1),...
                                      0:lbl_size(3):nDepth-lbl_size(3)-1); %for image data
    if prod(I_size) < 2^32-1
        cindex = uint32(cindex);
        rindex = uint32(rindex);
        dindex = uint32(dindex);
    end
    raw_indices = dindex(:)*nCols*nRows+cindex(:)*nRows+rindex(:);
    img_indices = bsxfun(@plus,img_ind(:),raw_indices(:)');
    clear cindex rindex dindex raw_indices;
    
    if processLabel
        [segRows, segCols, segDepth, ~] = size(pad_lbl);
        [cindex,rindex,dindex] = meshgrid(0:lbl_size(2)-1,0:lbl_size(1)-1,0:lbl_size(3)-1);
        if prod(I_size) < 2^32-1
            cindex = uint32(cindex);
            rindex = uint32(rindex);
            dindex = uint32(dindex);
        end
        seg_ind = dindex(:)*segCols*segRows+cindex(:)*segRows+rindex(:);
        [cindex,rindex,dindex] = meshgrid(0:lbl_size(2):segCols-1,...
                                          1:lbl_size(1):segRows,...
                                          0:lbl_size(3):segDepth-1); %for image data
        if prod(I_size) < 2^32-1
            cindex = uint32(cindex);
            rindex = uint32(rindex);
            dindex = uint32(dindex);
        end
        raw_indices = dindex(:)*segCols*segRows+cindex(:)*segRows+rindex(:);
        seg_indices = bsxfun(@plus,seg_ind(:),raw_indices(:)');
        clear cindex rindex dindex raw_indices;
    end

    % Generate the output
    Ix = zeros(im_size(1),im_size(2),im_size(3),I_size(4),size(img_indices,2),I_dtype);
    batches = 1:10:size(img_indices,2);
    for i = 1:I_size(4)
        for j = batches
            batch = j:min(j+9,size(img_indices,2));
            if isempty(batch)
                continue;
            end
            Ix(:,:,:,i,batch) = reshape(pad_img(img_indices(:,batch)+nDepth*nRows*nCols*(i-1)),im_size(1),im_size(2),im_size(3),1,numel(batch));
        end
    end
    
    if processLabel
        Jx = zeros(lbl_size(1),lbl_size(2),lbl_size(3),J_size(4),size(seg_indices,2),J_dtype);
%         for i = 1:J_size(4)
%             Jx(:,:,:,i,:) = reshape(pad_lbl(seg_indices+segDepth*segRows*segCols*(i-1)),lbl_size(1),lbl_size(2),lbl_size(3),1,size(seg_indices,2));
%         end
%         
        for j = batches
            batch = j:min(j+9,size(seg_indices,2));
            if isempty(batch)
                continue;
            end
            Jx(:,:,:,i,batch) = reshape(pad_lbl(seg_indices(:,batch)+segDepth*segRows*segCols*(i-1)),lbl_size(1),lbl_size(2),lbl_size(3),1,numel(batch));
        end
    end
    
end