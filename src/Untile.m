function [J] = Untile(Jx,sizes)
%UNTILE Summary of this function goes here
%   Detailed explanation goes here

    switch nargin
        case 1
            error('Need at least two inputs.')
        case 2
            im_size = sizes(1:2);
        otherwise
            error('Incorrect number of inputs.')
    end
    
    gridxy = ceil(im_size ./ [size(Jx,1),size(Jx,2)]);
    J = zeros(gridxy(1)*size(Jx,1),gridxy(2)*size(Jx,2),size(Jx,3),'like',Jx);
    img_ind = 0;
    ind_r = 1:size(Jx,1);
    ind_c = 1:size(Jx,2);
    for c = 1:gridxy(2)
        for r = 1:gridxy(1)
            img_ind = img_ind + 1;
            try
                J(ind_r + size(Jx,1)*(r-1), ind_c + size(Jx,2)*(c-1),:) = Jx(:,:,:,img_ind);
            catch err
                rethrow(err)
            end
        end
    end
    
    unpad = ([size(J,1),size(J,2)] - im_size)./2;
    J = J(floor(unpad(1))+1:im_size(1)+floor(unpad(1)),floor(unpad(2))+1:im_size(2)+floor(unpad(2)),:); 
end