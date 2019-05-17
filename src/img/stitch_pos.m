function [positions,im_sizes,correlation] = stitch_pos(im)
% stitch_pos.m
% Use generalized cross correlation to get stitch positions. The variable
% im should be a cell with the relative locations of where the images
% should be. For example, if a 2x2 set of images needs to be stitched
% together, then im should a 2x2 cell with the images placed in the
% relative positions of where they should be placed.
%
% This function returns a set of positions [row,col] as a cell of the same
% size as im. im_sizes is the size of the individual images, and f_size is
% the size of the final image after all images are put together.
%
% TODO:
% Currently, if there are images that cannot be stitched together then the
% relative position of the images is used to calculate all unknown
% positions. This should be changed so that all images with unknown
% positions that border each other are correlated to see if they can be
% locally placed with respect to each other. This should improve stitching
% accuracy.

% image sizes
[nRows,nCols] = cellfun(@size,im(:));
im_sizes = mat2cell([nRows nCols],ones(numel(nRows),1),2);
im_sizes = reshape(im_sizes,size(im));

% initialize seek matrices
known_pos = false(size(im));
known_pos(round(end/2),round(end/2)) = true;
analyzed_once = known_pos;
needs_analysis = known_pos;
neighbors = bsxfun(@plus,1:numel(im),[-1;size(im,1);1;-size(im,1)]);
neighbors(1,1:size(im,1):numel(im)) = 0;
neighbors(3,size(im,1):size(im,1):numel(im)) = 0;
neighbors(4,1:size(im,1)) = 0;
neighbors(2,end-size(im,1)-1:end) = 0;

% initialize outputs
positions = cell(size(im));
positions{known_pos} = [0,0];
correlation = zeros(size(im));
overlap = correlation;

disp('Stitching image...')
disp('0%')
per_com = 0;
while any(needs_analysis(:))
    pos = find(needs_analysis(:))';
    for p = pos
        for i = 1:4
            % Skip invalid positions
            t = neighbors(i,p);
            if t==0 || known_pos(t)
                continue;
            end
            
            if isempty(im{t})
                needs_analysis(t) = false;
                known_pos(t) = true;
                continue;
            end
                        
            % find the cross correlation
            [C,o] = normxcorr2_general(im{t},im{p},round(0.01*prod(im_sizes{p})));
            [~,index] = max(C(:));
            correlation(t) = C(index);
            overlap(t) = o(index)/prod(im_sizes{p});
            
            % if cross-correlation is greater than 0.9 and overlap is
            % roughly 10%, then we have a match
            analyzed_once(t) = true;
            if correlation(t)>0.9 && overlap(t)<0.09 && overlap(t)>0.01
                known_pos(t) = true;
                needs_analysis(t) = true;
                [row, col] = ind2sub(size(C),index);
                positions{t} = [row,col] - im_sizes{p} + positions{p};
            end
        end
        needs_analysis(p) = false;
    end
    per_com = ceil(100*sum(known_pos(:))/numel(known_pos));
    disp([num2str(per_com) '%'])
end

for i = 1:numel(positions)
    if isempty(positions{i})
        positions{i} = [0 0];
    end
end

% Calculate average translation in x and y directions for all known
% positions
row_trans = cell(size(positions));
col_trans = cell(size(positions));
for i = 1:numel(positions)
    row_trans{i} = [0 0];
    col_trans{i} = [0 0];
    row_sum = sum(known_pos(neighbors(find(neighbors([1 3],i))*2-1,i)));
    col_sum = sum(known_pos(neighbors(find(neighbors([2 4],i))*2,i)));
    for j = 1:2:3
        if neighbors(j,i)~=0
            row_trans{i} = row_trans{i} + double(known_pos(neighbors(j,i))) * ...
                           (j-2) * double(neighbors(j,i)>0) *...
                           (positions{neighbors(j,i)} - positions{i})  ./...
                           row_sum;
        end
        if neighbors(j+1,i)~=0
            col_trans{i} = col_trans{i} + double(known_pos(neighbors(j+1,i))) * ...
                           (j-2) * double(neighbors(j+1,i)>0) *...
                           (positions{i} - positions{neighbors(j+1,i)}) ./...
                           col_sum;
        end
    end
end

mean_col_trans = nanmean(reshape([col_trans{known_pos(:)}],2,[])');
mean_row_trans = nanmean(reshape([row_trans{known_pos(:)}],2,[])');

% Find images that have no known position and place them in the image
% according to their relative location in the stitched image
[r,c] = find(~known_pos);
r_offset = round(size(known_pos,1)/2);
c_offset = round(size(known_pos,2)/2);
for i = 1:numel(r)
    positions{r(i),c(i)} = round((r(i)-r_offset).*mean_row_trans +...
                           (c(i)-c_offset).*mean_col_trans);
end

nRows = reshape(nRows,size(im));

positions = mat2cell(cat(1,positions{:}) - min(cat(1,positions{:})) + 1,ones(numel(nRows),1),2);
positions = reshape(positions,size(im));

end