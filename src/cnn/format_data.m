function [imdb] = format_data(opts)
% FORMAT_DATA pre-processes data for CNN training
% This file has been modified for CADASIL cerebral capillary analysis.
% NJS - May 3, 2018

    %% Parse inputs and determine if data has been previously formatted
    if ~exist([opts.expDir ' 1'],'dir')
        mkdir([opts.expDir ' 1']);
    elseif exist(fullfile([opts.expDir ' 1'],'imdb.mat'),'file')
        disp('Data already exists. Loading and checking data...')
        load(fullfile([opts.expDir ' 1'],'imdb.mat'));
        return
    end
    
    %% Initialize the image database
    imdb = struct();
    
    if opts.prep.preLoad
        imdb.images.data = single([]); % images
        imdb.images.label = single([]); % label for segment
        imdb.images.weight = single([]);
    else
        imdb.images.data = {}; % images
        status = mkdir(fullfile(opts.expDir,'data'));
        imdb.images.label = {}; % label for segment
        status = mkdir(fullfile(opts.expDir,'label'));
        imdb.images.weight = {};
        status = mkdir(fullfile(opts.expDir,'weight'));
    end

    imdb.images.set = []; % 1 if image is for training, 2 if image is for validation
    imdb.images.imgFile = {}; % numerical id for image
    imdb.images.lblFile = {};

    %% Check if directories are valid, get images
    if ~exist(opts.imgPath)
        error('Invalid raw image input directory.')
    elseif ~exist(opts.lblPath)
        error('Invalid segmented image input directory.')
    end
    
    raw_images = dir(fullfile(opts.imgPath,'*.tif*'));
    region_images = dir(fullfile(opts.lblPath,'Region Segmentation','*.tif*'));
    vessel_images = dir(fullfile(opts.lblPath,'Vessel Segmentation','*.tif*'));
    
    %% Load files and format data
    disp('-----------------------------------------------')
    disp(' Loading and Formating')
    disp('-----------------------------------------------')
    
    if opts.loadParallel
        p = gcp();
        raw_par = parfeval(p,@imread,1,fullfile(opts.imgPath,raw_images(1).name));
        seg_par = parfeval(p,@imread,1,fullfile(opts.lblPath,raw_images(1).name));
    end
    
    for i = 1:length(raw_images)
        % Load raw and segmented images
        disp(['Raw Image ' num2str(i) ': ' raw_images(i).name])
        if opts.loadParallel
            raw_img = fetchOutputs(raw_par);
            if i < length(raw_images)
                raw_par = parfeval(p,@imread,1,fullfile(opts.imgPath,raw_images(i+1).name));
            end
        else
            raw_img = imread(fullfile(opts.imgPath,raw_images(i).name));
        end
        raw_img = single(raw_img);
        
        disp(['Brain Region Segmentation Image ' num2str(i) ': ' region_images(i).name]);
        disp(['Brain Vessel Segmentation Image ' num2str(i) ': ' vessel_images(i).name]);
        if opts.loadParallel
            error('This code was not designed for parallel loading of images.')
%             seg_img = fetchOutputs(seg_par);
%             if i < length(raw_images)
%                 seg_par = parfeval(p,@imread,1,fullfile(opts.lblPath,region_images(i+1).name));
%             end
        else
            region_img = imread(fullfile(opts.lblPath,'Region Segmentation',region_images(i).name));
            vessel_img = imread(fullfile(opts.lblPath,'Vessel Segmentation',vessel_images(i).name));
        end
        region_img = single(region_img);
        vessel_img = single(vessel_img>0);
        vessel_img(vessel_img==0) = -1;
        
        if size(raw_img,1)~=size(region_img,1) || size(raw_img,2)~=size(region_img,2)
            disp('Raw and segmented images are not the same size. Moving to next image.')
            disp('')
            continue
        end
        disp(' ');
        imdb.images.imgFile(end+1) = {fullfile(opts.imgPath,raw_images(i).name)};
        imdb.images.lblFile(end+1) = {{fullfile(opts.lblPath,'Region Segmentation',region_images(i).name),...
                                       fullfile(opts.lblPath,'Vessel Segmentation',region_images(i).name)}};
        
        % Normalize local pixel intensity
        if opts.prep.normImage
            norm = MaskedLocalResponse(raw_img,raw_img>0,opts.prep.windowSize);
            norm(norm>opts.prep.maxNorm) = opts.prep.maxNorm;
            norm(norm<-opts.prep.maxNorm) = -opts.prep.maxNorm;
            raw_img = norm;
        end
        
        % Create weights based on distance from boundary
        weights = ones([size(region_img),2]);
        % Region Segmentation Weights
        num_regions = numel(unique(region_img));
        weight_count = zeros(num_regions,1);
        region_weights = weights(:,:,1);
        for j = 1:num_regions
            r = DistanceResponse(region_img==j,opts.prep.foreWeightThresh);
            region_weights(region_img==j) = r(region_img==j).^opts.prep.foreWeightAmp;
            weight_count(j) = sum(double(region_weights(region_img==j)));
        end
        m = max(region_weights(:));
        for j = 1:num_regions
            region_weights(region_img==j) = region_weights(region_img==j).*(sum(weight_count)/weight_count(j))./m;
        end
        weights(:,:,1) = region_weights;
        % Vessel Segmentation Weights
        vessel_weights = weights(:,:,2);
        r = DistanceResponse(vessel_img==-1,10);
        vessel_weights(vessel_img==-1) = r(vessel_img==-1).^-1;
        vessel_weights(vessel_img==1) = sum(vessel_weights(vessel_img==-1))/sum(vessel_weights(vessel_img==1));
        weights(:,:,2) = vessel_weights;
        %[raw_img,region_img,weights] = Weight(raw_img,region_img);
        
        % Generate indices
        [nRows, nCols, ~] = size(raw_img);
        [cindex,rindex] = meshgrid(0:opts.prep.imgSize(2)-1,0:opts.prep.imgSize(1)-1);
        img_ind = cindex(:)*nRows+rindex(:);
        seg_offsets = (opts.prep.imgSize-opts.prep.lblSize)/2;
        [cindex,rindex] = meshgrid(seg_offsets(2):opts.prep.imgSize(2)-seg_offsets(2)-1,...
                                   seg_offsets(1):opts.prep.imgSize(1)-seg_offsets(1)-1);
        seg_ind = cindex(:)*nRows+rindex(:);
        
        [cindex,rindex] = meshgrid(0:opts.prep.lblSize(2):nCols-opts.prep.imgSize(2),...
                                   1:opts.prep.lblSize(1):nRows-opts.prep.imgSize(1)+1); %for image data
        raw_indices = nRows*cindex(:)+rindex(:);
        img_indices = bsxfun(@plus,img_ind(:),raw_indices(:)');
        seg_indices = bsxfun(@plus,seg_ind(:),raw_indices(:)');
        
        % Get pixels and reshape into images
        img_pixels = [];
        for j = 1:length(opts.prep.channels)
            img = raw_img(:,:,opts.prep.channels(j));
            img_chan_pix = img(img_indices(:));
            img_chan_pix = reshape(img_chan_pix,opts.prep.imgSize(1),opts.prep.imgSize(2),1,[]);
            img_pixels = cat(3,img_pixels,img_chan_pix);
        end
        
        % get the region images
        seg_pixels = region_img(seg_indices(:));
        seg_pixels = reshape(seg_pixels,opts.prep.lblSize(1),opts.prep.lblSize(2),1,[]);
        weight_pixels = weights(:,:,1);
        weight_pixels = weight_pixels(seg_indices(:));
        weight_pixels = reshape(weight_pixels,opts.prep.lblSize(1),opts.prep.lblSize(2),1,[]);
        
        % get the vessel segmentation images
        vessel_pixels = vessel_img(seg_indices(:));
        vessel_pixels = reshape(vessel_pixels,opts.prep.lblSize(1),opts.prep.lblSize(2),1,[]);
        seg_pixels = cat(3,vessel_pixels,seg_pixels);
        vessel_weights = weights(:,:,2);
        vessel_weights = vessel_weights(seg_indices(:));
        vessel_weights = reshape(vessel_weights,opts.prep.lblSize(1),opts.prep.lblSize(2),1,[]);
        weight_pixels = cat(3,weight_pixels,vessel_weights);
        
        % Add images to image database
        if opts.prep.preLoad
            imdb.images.data = cat(4,imdb.images.data,img_pixels);
            imdb.images.label = single(cat(4,imdb.images.label, seg_pixels));
            imdb.images.weight = single(cat(4,imdb.images.weight, weight_pixels));            
        else
            error('Training this network requires preloading the data.')
        end

    end
    
    disp(['Formatted ' num2str(size(imdb.images.data,4)) ' images!']);
    if isempty(imdb.images.data)
        return
    end
        
    % Reset the random number generator
    rng(opts.train.randomSeed);
    setIndex = randsample(size(imdb.images.label,4),round(size(imdb.images.label,4)*opts.prep.holdout));
    imdb.images.set = single(ones(size(imdb.images.label,4),1));
    imdb.images.set(setIndex) = 2;

end

