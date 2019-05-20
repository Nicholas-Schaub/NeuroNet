function [S] = CNNAttribute(img,net,opts,batchSize,useGpu)
% segment.m
%
% This function uses a convolutional neural network trained using
% MatConvNet to segment an image. Edges of the image are segmented by
% padding the image using reflection. It is assumed that the size of the
% input image can take on any power of 2. It is also assumed that any
% cropping caused by the networks segmentation is constant regardless of
% image size, and that cropping is uniform in both the horizontal and
% vertical axes.
%
% The third input of the function is how much padding should be applied to
% the edges of the image. If no value is given, this value is estimated by
% segmenting a small portion of the image. The third input is expected to
% be a scalar.
%
% The img input can be a cell array of images to be segmented, provided all
% images are the same size.

%% Initialize the network
    tic;
    switch class(net)
        case 'dagnn.DagNN' % matconvnet dag network
            isMCN = true;
        case 'DAGNetwork' % matlab dag network
            isMCN = false;
            pred_ind = 'auto';
        case 'struct' % structure of matconvnet dag network
            isMCN = true;
            net = dagnn.DagNN.loadobj(net);
        case 'nnet.cnn.LayerGraph' % layergraph of matlab dag network
            net = mcn2mat(net,opts);
            pred_ind = 'auto';
        otherwise
            error('Could not determine the type of neural network from the 2nd input (net).')
    end
    
    nn = net;
    

    
    if useGpu
        env = 'gpu';
    else
        env = 'cpu';
    end
        
    if isMCN
        %net.mode = 'test';
        pred_ind = net.getVarIndex('pred');
        lbl_ind = net.getVarIndex('label');
        if isnan(pred_ind)
            error('Could not find prediction layer.');
        end
        if isnan(lbl_ind)
            error('Could not find label layer.');
        end
        nn.vars(pred_ind).precious = 1;
        nn.vars(lbl_ind).precious = 1;
    end
    disp(['Time to load network: ' num2str(toc)])

%% Normalize and Pad the Image
    tic;
    img = single(img);
    if opts.prep.normImage
        img = Preprocess(img,opts);
    end
    
    img_pixels = Tile(img,opts);
    
    if useGpu
        img_pixels = gpuArray(img_pixels);
    end
    disp(['Time to pad and tile the image: ' num2str(toc)]);
        
%% Segment the image
    tic;
    
    if isMCN
        sub_batches = 0:batchSize:size(img_pixels,4);
        if sub_batches(end) ~= size(img_pixels,4)
            sub_batches(end+1) = size(img_pixels,4);
        end

        seg_img = zeros(opts.prep.lblSize(1),opts.prep.lblSize(2),1,sub_batches(end),'like',nn.params(1).value);

        for batch = 1:length(sub_batches)-1
            nn.eval({'input',img_pixels(:,:,1,sub_batches(batch)+1:sub_batches(batch+1)),'label',seg_img(:,:,1,sub_batches(batch)+1:sub_batches(batch+1))});
            if useGpu
                seg_img(:,:,1,sub_batches(batch)+1:sub_batches(batch+1)) = gather(nn.getVar('pred').value);
            else
                seg_img(:,:,1,sub_batches(batch)+1:sub_batches(batch+1)) = (nn.getVar('pred').value);
            end
            disp(['Finished batch ' num2str(batch) ' of ' num2str(length(sub_batches)-1) '. Total duration: ' num2str(toc)]);
        end
    else
        seg_img = predict(nn,img_pixels,'MiniBatchSize',batchSize);
    end
    disp(['Time to segment the image: ' num2str(toc)])
    
%% Construct segmented image
    tic;
    S = Untile(seg_img,size(img));
    disp(['Time to reconstruct image: ' num2str(toc)]);
end