function [imdb] = format_data_nuclei(opts)
% FORMAT_DATA pre-processes data for CNN training

    %% Parse inputs and determine if data has been previously formatted
%     if ~exist([opts.expDir ' 1'],'dir')
%         mkdir([opts.expDir ' 1']);
%     elseif exist(fullfile([opts.expDir ' 1'],'imdb.mat'),'file')
%         disp('Data already exists. Loading and checking data...')
%         load(fullfile([opts.expDir ' 1'],'imdb.mat'));
%         return
%     end
    
    %% Preprocess datasets
    %  Make sure all data is in the same format.
    
    % paths to all preprocessed images
    raw_images = cell(0,0);
    seg_images = cell(0,0);
    n_total = 0;
    
    %% Data from Koyuncu et al
    %  http://www.cs.bilkent.edu.tr/~gunduz/downloads/NucleusSegData/
    %  # of images: 37
    %  # of exluded images: 0
    %  # of network images: 444
    %  # of cells : 2661
    %  Summary: 
    
    disp('Preprocessing Koyuncu:      ');
    
    % get information about the quantity of data
    n_images = 0;
    n_cells = 0;
    n_subimages = 0;
    n_excluded = 0;
    
    % setup the image paths
    im_path = {fullfile(opts.imgPath,'Koyuncu','Huh7TestSet'),...
               fullfile(opts.imgPath,'Koyuncu','HepG2TestSet'),...
               fullfile(opts.imgPath,'Koyuncu','TrainingSet')};
    im_out = fullfile(opts.imgPath,'Koyuncu','images');
    lbl_out = fullfile(opts.imgPath,'Koyuncu','labels');
    mkdir(im_out);
    mkdir(lbl_out);
    
    % Loop through images
    d = dir(fullfile(im_path{1},'*.jpg'));
        
    for i = 1:numel(d)
        
        % if images have been processed, then skip
        if exist(fullfile(im_out,strrep(d(i).name,'jpg','tif')),'file') && ...
           exist(fullfile(lbl_out,strrep(d(i).name,'jpg','tif')),'file')
            raw_images{end+1} = fullfile(im_out,strrep(d(i).name,'jpg','tif'));
            seg_images{end+1} = fullfile(lbl_out,strrep(d(i).name,'jpg','tif'));
            fprintf('\b\b\b\b\b\b%05.2f%%', i/numel(d)*100);
            continue;
        end
        
        % load image and save nuclear channel
        I = imread(fullfile(d(i).folder,d(i).name));
        n_images = n_images + 1;
        subimages = floor(size(I,1)/200)*floor(size(I,2)/200);
        if subimages == 0
            subimages = 1;
        end
        n_subimages = n_subimages + subimages;
        imwrite(I(:,:,3),fullfile(im_out,strrep(d(i).name,'jpg','tif')),'compression','deflate');
        raw_images{end+1} = fullfile(im_out,strrep(d(i).name,'jpg','tif'));
        
        % load labels and save
        S = regexp(d(i).name,'huh7_ts([\d]+).jpg','tokens');
        I = readmatrix(fullfile(d(i).folder,['huh7_ts_ann' S{1}{1}]));
        I = I(1:768,1:1024);
        n_cells = n_cells + double(max(I(:)));
        L = separate_nuclei(I);
        imwrite(L>0,fullfile(lbl_out,strrep(d(i).name,'jpg','tif')),'compression','deflate');
        seg_images{end+1} = fullfile(lbl_out,strrep(d(i).name,'jpg','tif'));
        
        fprintf('\b\b\b\b\b\b%05.2f%%', i/37*100);

    end
    
    d = dir(fullfile(im_path{2},'*.jpg'));
    for i = 1:numel(d)
        
        % if images have been processed, then skip
        if exist(fullfile(im_out,strrep(d(i).name,'jpg','tif')),'file') && ...
           exist(fullfile(lbl_out,strrep(d(i).name,'jpg','tif')),'file')
            raw_images{end+1} = fullfile(im_out,strrep(d(i).name,'jpg','tif'));
            seg_images{end+1} = fullfile(lbl_out,strrep(d(i).name,'jpg','tif'));
            fprintf('\b\b\b\b\b\b%05.2f%%', i/numel(d)*100);
            continue;
        end
        
        % load image and save nuclear channel
        I = imread(fullfile(d(i).folder,d(i).name));
        n_images = n_images + 1;
        subimages = floor(size(I,1)/200)*floor(size(I,2)/200);
        if subimages == 0
            subimages = 1;
        end
        n_subimages = n_subimages + subimages;
        imwrite(I(:,:,3),fullfile(im_out,strrep(d(i).name,'jpg','tif')),'compression','deflate');
        raw_images{end+1} = fullfile(im_out,strrep(d(i).name,'jpg','tif'));
        
        % load labels and save
        S = regexp(d(i).name,'hepg2_ts([\d]+).jpg','tokens');
        I = readmatrix(fullfile(d(i).folder,['hepg2_ts_ann' S{1}{1}]));
        I = I(1:768,1:1024);
        n_cells = n_cells + double(max(I(:)));
        L = separate_nuclei(I);
        imwrite(L>0,fullfile(lbl_out,strrep(d(i).name,'jpg','tif')),'compression','deflate');
        seg_images{end+1} = fullfile(lbl_out,strrep(d(i).name,'jpg','tif'));
        
        fprintf('\b\b\b\b\b\b%05.2f%%', (i+11)/37*100);

    end
    
    d = dir(fullfile(im_path{3},'*.jpg'));
    for i = 1:numel(d)
        
        % if images have been processed, then skip
        if exist(fullfile(im_out,strrep(d(i).name,'jpg','tif')),'file') && ...
           exist(fullfile(lbl_out,strrep(d(i).name,'jpg','tif')),'file')
            raw_images{end+1} = fullfile(im_out,strrep(d(i).name,'jpg','tif'));
            seg_images{end+1} = fullfile(lbl_out,strrep(d(i).name,'jpg','tif'));
            fprintf('\b\b\b\b\b\b%05.2f%%', i/numel(d)*100);
            continue;
        end
        
        % load image and save nuclear channel
        I = imread(fullfile(d(i).folder,d(i).name));
        n_images = n_images + 1;
        subimages = floor(size(I,1)/200)*floor(size(I,2)/200);
        if subimages == 0
            subimages = 1;
        end
        n_subimages = n_subimages + subimages;
        imwrite(I(:,:,3),fullfile(im_out,strrep(d(i).name,'jpg','tif')),'compression','deflate');
        raw_images{end+1} = fullfile(im_out,strrep(d(i).name,'jpg','tif'));
        
        % load labels and save
        S = regexp(d(i).name,'tr([\d]+).jpg','tokens');
        I = readmatrix(fullfile(d(i).folder,['tr_ann' S{1}{1}]));
        I = I(1:768,1:1024);
        n_cells = n_cells + double(max(I(:)));
        L = separate_nuclei(I);
        imwrite(L>0,fullfile(lbl_out,strrep(d(i).name,'jpg','tif')),'compression','deflate');
        seg_images{end+1} = fullfile(lbl_out,strrep(d(i).name,'jpg','tif'));
        
        fprintf('\b\b\b\b\b\b%05.2f%%', (i+27)/37*100);

    end
    
    fprintf('\b\b\b\b\b\b\b100.00%%');
    warning('on');
    fprintf('\n');
    disp(['Number of images: ' num2str(n_images)]);
    disp(['Number of excluded images: ' num2str(n_excluded)]);
    disp(['Number of network images: ' num2str(n_subimages)]);
    disp(['Number of cells: ' num2str(n_cells)]);
    disp(' ');
    n_total = n_total + n_images;
    
    %% Data from Broad Institute (BBBC039)
    %  https://data.broadinstitute.org/bbbc/BBBC039/
    %  Note: According to the Broad Institute website, there is some
    %  overlap in the images in this dataset with the Kaggle dataset. It
    %  was not determined which images were overlapping between the
    %  datasets, so there will be some repeats in the total data.
    %  # of images: 200
    %  # of exluded images: 0
    %  # of network images: 800
    %  # of cells : 19565
    %  Summary: 
    
    disp('Preprocessing BBBC008:      ');
    
    % get information about the quantity of data
    n_images = 0;
    n_cells = 0;
    n_subimages = 0;
    n_excluded = 0;
    
    % setup the image paths
    im_path = fullfile(opts.imgPath,'BBBC039','data');
    lbl_path = fullfile(opts.imgPath,'BBBC039','masks');
    im_out = fullfile(opts.imgPath,'BBBC039','images');
    lbl_out = fullfile(opts.imgPath,'BBBC039','labels');
    mkdir(im_out);
    mkdir(lbl_out);
    
    % Loop through images
    d = dir(fullfile(im_path,'*.tif'));
    l = dir(fullfile(lbl_path,'*.png'));
        
    for i = 1:numel(d)
        
        % if images have been processed, then skip
        if exist(fullfile(im_out,d(i).name),'file') && ...
           exist(fullfile(lbl_out,d(i).name),'file')
            raw_images{end+1} = fullfile(im_out,d(i).name);
            seg_images{end+1} = fullfile(lbl_out,d(i).name);
            fprintf('\b\b\b\b\b\b%05.2f%%', i/numel(d)*100);
            continue;
        end
        
        % load image and save nuclear channel
        I = imread(fullfile(d(i).folder,d(i).name));
        n_images = n_images + 1;
        subimages = floor(size(I,1)/200)*floor(size(I,2)/200);
        if subimages == 0
            subimages = 1;
        end
        n_subimages = n_subimages + subimages;
        imwrite(I(:,:,1),fullfile(im_out,d(i).name),'compression','deflate');
        raw_images{end+1} = fullfile(im_out,d(i).name);
        
        % load labels and save
        I = imread(fullfile(l(i).folder,strrep(d(i).name,'tif','png')));
        L = bwlabel(I(:,:,1)>0);
        n_cells = n_cells + double(max(L(:)));
        imwrite(L>0,fullfile(lbl_out,d(i).name),'compression','deflate');
        seg_images{end+1} = fullfile(lbl_out,d(i).name);
        
        fprintf('\b\b\b\b\b\b%05.2f%%', i/numel(d)*100);

    end
    fprintf('\b\b\b\b\b\b\b100.00%%');
    warning('on');
    fprintf('\n');
    disp(['Number of images: ' num2str(n_images)]);
    disp(['Number of excluded images: ' num2str(n_excluded)]);
    disp(['Number of network images: ' num2str(n_subimages)]);
    disp(['Number of cells: ' num2str(n_cells)]);
    disp(' ');
    n_total = n_total + n_images;
        
    %% Kaggle 2018 cell nuclei competition (aka BBB038)
    %  https://www.kaggle.com/c/data-science-bowl-2018/overview
    %  Corrections to stage 1 hand labeling: https://github.com/ibmua/data-science-bowl-2018-train-set
    %  Broad Institute link: https://data.broadinstitute.org/bbbc/BBBC038/
    %  # of images: 683
    %  # of excluded images: 3071 (excluded if not fluorescent or no ground truth
    %  # of network images: 1273
    %  # of cells : 29,268
    %  image sizes: variable, some are smaller than 256x256
    %  Summary: 1. Images are RGB, so extract only nuclear channel.
    %           2. Labels are RLE encoded, so decompress and save image.
    
    disp('Preprocessing Kaggle:      ');
    
    % get information about the quantity of data
    n_images = 0;
    n_cells = 0;
    n_subimages = 0;
    n_excluded = 0;
    
    % setup the image paths
    im_path = fullfile(opts.imgPath,'Kaggle','extracted');
    im_out = fullfile(opts.imgPath,'Kaggle','images');
    lbl_out = fullfile(opts.imgPath,'Kaggle','labels');
    mkdir(im_path);
    mkdir(im_out);
    mkdir(lbl_out);
    
    % load the image labels
    im_lbl = readtable(fullfile(opts.imgPath,'Kaggle','stage2_solution_final.csv'));
    im_temp = readtable(fullfile(opts.imgPath,'Kaggle','stage1_solution.csv'));
    im_lbl = [im_lbl; im_temp];
    
    % Loop through the test images
    d = dir([im_path]); % get image folders
    warning('off'); % one png file generates a warning that does not inhibit loading
    for i = 3:numel(d)
        
        % if images have been processed, then skip
        if exist(fullfile(im_out,[d(i).name '.tif']),'file') && ...
           exist(fullfile(lbl_out,[d(i).name '.tif']),'file')
            raw_images{end+1} = fullfile(fullfile(im_out,[d(i).name '.tif']));
            seg_images{end+1} = fullfile(fullfile(lbl_out,[d(i).name '.tif']));
            fprintf('\b\b\b\b\b\b%05.2f%%', i/numel(d)*100);
            continue;
        end
        
        % load image and save nuclear channel
        I = imread(fullfile(d(i).folder,d(i).name,'images',[d(i).name '.png']));
        if mean(I(:))>100 % some of the image are not fluorescent, so exclude them
            n_excluded = n_excluded + 1;
            continue;
        end
        I = I(:,:,1);
        
        % If mask exists, use that to generate the image. Otherwise, do RLE
        % decompression
        if exist(fullfile(d(i).folder,d(i).name,'masks'),'dir')
            % Build image from corrected images
            n_images = n_images + 1;
            subimages = floor(size(I,1)/200)*floor(size(I,2)/200);
            if subimages == 0
                subimages = 1;
            end
            n_subimages = n_subimages + subimages;
            l = dir(fullfile(d(i).folder,d(i).name,'masks','*.png'));
            L = zeros(size(I));
            try
                L = double(imread(fullfile(l(1).folder,l(1).name))>0);
            catch err
                if isempty(l)
                    L = zeros(size(I));
                else
                    rethrow(err);
                end
            end
            for j = 2:numel(l)
                L = L + double(imread(fullfile(l(j).folder,l(j).name))>0).*(j-2);
            end
        else
            % RLE decompression
            L = zeros(size(I,1),size(I,2));                   % initialize output
            nuc_ind = find(strcmp(im_lbl.ImageId,d(i).name)); % find indices of nuclei
            if numel(nuc_ind)==1
                pix_ind = str2num(im_lbl.EncodedPixels{nuc_ind(1)});
                if numel(pix_ind)==2 && pix_ind(1)==1 && pix_ind(2)==1
                    n_excluded = n_excluded + 1;
                    continue;
                end
            end
            n_images = n_images + 1;
            subimages = floor(size(I,1)/200)*floor(size(I,2)/200);
            if subimages == 0
                subimages = 1;
            end
            n_subimages = n_subimages + subimages;
            for j = 1:numel(nuc_ind)                          % loop through nuclei
                pix_ind = str2num(im_lbl.EncodedPixels{nuc_ind(j)});
                for k = 1:2:numel(pix_ind)
                    L(pix_ind(k) + (0:pix_ind(k+1)-1)) = L(pix_ind(k) + (0:pix_ind(k+1)-1)) + j;
                end
            end
        end
        
        % some nuclei touch, so make sure that touching borders are zeroed
        L = separate_nuclei(L);
        n_cells = n_cells + double(max(L(:)));
        if max(L(:))
            warning('No cells detected.');
        end
        L = bwareafilt(L>0,[15 Inf]);
        L = imfill(L,'holes');
        imwrite(I,fullfile(im_out,[d(i).name '.tif']),'compression','deflate');
        raw_images{end+1} = fullfile(im_out,[d(i).name '.tif']);
        imwrite(L,fullfile(lbl_out,[d(i).name '.tif']),'compression','deflate');
        seg_images{end+1} = fullfile(lbl_out,[d(i).name '.tif']);
        
        fprintf('\b\b\b\b\b\b%05.2f%%', i/numel(d)*100);
    end
    fprintf('\b\b\b\b\b\b\b100.00%%');
    warning('on');
    fprintf('\n');
    disp(['Number of images: ' num2str(n_images)]);
    disp(['Number of excluded images: ' num2str(n_excluded)]);
    disp(['Number of network images: ' num2str(n_subimages)]);
    disp(['Number of cells: ' num2str(n_cells)]);
    disp(' ');
    n_total = n_total + n_images;
    
    %% Data from Broad Institute (BBBC020)
    %  https://data.broadinstitute.org/bbbc/BBBC020/
    %  # of images: 20
    %  # of exluded images: 5
    %  # of network images: 400
    %  # of cells : 8831
    %  Summary: 
    
    disp('Preprocessing BBBC020:      ');
    
    % get information about the quantity of data
    n_images = 0;
    n_cells = 0;
    n_subimages = 0;
    n_excluded = 0;
    
    % setup the image paths
    im_path = fullfile(opts.imgPath,'BBBC020','BBBC020_v1_images');
    lbl_path = fullfile(opts.imgPath,'BBBC020','BBC020_v1_outlines_nuclei');
    im_out = fullfile(opts.imgPath,'BBBC020','images');
    lbl_out = fullfile(opts.imgPath,'BBBC020','labels');
    mkdir(im_out);
    mkdir(lbl_out);
    
    % Loop through images
    d = dir(im_path);
    l = dir(lbl_path);
        
    for i = 3:numel(d)
        
        % if images have been processed, then skip
        if exist(fullfile(im_out,[d(i).name '.tif']),'file') && ...
           exist(fullfile(lbl_out,[d(i).name '.tif']),'file')
            raw_images{end+1} = fullfile(im_out,[d(i).name '.tif']);
            seg_images{end+1} = fullfile(lbl_out,[d(i).name '.tif']);
            fprintf('\b\b\b\b\b\b%05.2f%%', i/numel(d)*100);
            continue;
        end
        
        % load image and save nuclear channel
        I = imread(fullfile(d(i).folder,d(i).name,[d(i).name '_c5.TIF']));
        
        % load image and save nuclear channel
        ind = find(startsWith({l.name},d(i).name));
        try
            L = double(imread(fullfile(l(ind(1)).folder,l(ind(1)).name))>0);
        catch err
            n_excluded = n_excluded + 1;
            continue;
        end
        imwrite(I(:,:,3),fullfile(im_out,[d(i).name '.tif']),'compression','deflate');
        raw_images{end+1} = fullfile(im_out,[d(i).name '.tif']);
        n_images = n_images + 1;
        subimages = floor(size(I,1)/200)*floor(size(I,2)/200);
        if subimages == 0
            subimages = 1;
        end
        n_subimages = n_subimages + subimages;
        for j = ind(2:end)
            L = L + double(imread(fullfile(l(j).folder,l(j).name))>0).*j;
        end
        n_cells = n_cells + double(max(L(:)));
        imwrite(L>0,fullfile(lbl_out,[d(i).name '.tif']),'compression','deflate');
        seg_images{end+1} = fullfile(lbl_out,[d(i).name '.tif']);
        
        fprintf('\b\b\b\b\b\b%05.2f%%', i/numel(d)*100);

    end
    fprintf('\b\b\b\b\b\b\b100.00%%');
    warning('on');
    fprintf('\n');
    disp(['Number of images: ' num2str(n_images)]);
    disp(['Number of excluded images: ' num2str(n_excluded)]);
    disp(['Number of network images: ' num2str(n_subimages)]);
    disp(['Number of cells: ' num2str(n_cells)]);
    disp(' ');
    n_total = n_total + n_images;
    
    %% Data from Broad Institute (BBBC008)
    %  https://data.broadinstitute.org/bbbc/BBBC008/
    %  # of images: 12
    %  # of exluded images: 0
    %  # of network images: 48
    %  # of cells : 1448
    %  Summary: 
    
    disp('Preprocessing BBBC008:      ');
    
    % get information about the quantity of data
    n_images = 0;
    n_cells = 0;
    n_subimages = 0;
    n_excluded = 0;
    
    % setup the image paths
    im_path = fullfile(opts.imgPath,'BBBC008','human_ht29_colon_cancer_2_images');
    lbl_path = fullfile(opts.imgPath,'BBBC008','human_ht29_colon_cancer_2_foreground');
    im_out = fullfile(opts.imgPath,'BBBC008','images');
    lbl_out = fullfile(opts.imgPath,'BBBC008','labels');
    mkdir(im_out);
    mkdir(lbl_out);
    
    % Loop through images
    d = dir(fullfile(im_path,'*channel1.tif'));
    l = dir(fullfile(lbl_path,'*channel1.tif'));
        
    for i = 1:numel(d)
        
        % if images have been processed, then skip
        if exist(fullfile(im_out,d(i).name),'file') && ...
           exist(fullfile(lbl_out,d(i).name),'file')
            raw_images{end+1} = fullfile(im_out,d(i).name);
            seg_images{end+1} = fullfile(lbl_out,d(i).name);
            fprintf('\b\b\b\b\b\b%05.2f%%', i/numel(d)*100);
            continue;
        end
        
        % load image and save nuclear channel
        I = imread(fullfile(d(i).folder,d(i).name));
        n_images = n_images + 1;
        subimages = floor(size(I,1)/200)*floor(size(I,2)/200);
        if subimages == 0
            subimages = 1;
        end
        n_subimages = n_subimages + subimages;
        imwrite(I(:,:,1),fullfile(im_out,d(i).name),'compression','deflate');
        raw_images{end+1} = fullfile(im_out,d(i).name);
        
        % load labels and save
        I = imread(fullfile(l(i).folder,d(i).name));
        L = bwlabel(I);
        n_cells = n_cells + double(max(L(:)));
        imwrite(I>0,fullfile(lbl_out,d(i).name),'compression','deflate');
        seg_images{end+1} = fullfile(lbl_out,d(i).name);
        
        fprintf('\b\b\b\b\b\b%05.2f%%', i/numel(d)*100);

    end
    fprintf('\b\b\b\b\b\b\b100.00%%');
    warning('on');
    fprintf('\n');
    disp(['Number of images: ' num2str(n_images)]);
    disp(['Number of excluded images: ' num2str(n_excluded)]);
    disp(['Number of network images: ' num2str(n_subimages)]);
    disp(['Number of cells: ' num2str(n_cells)]);
    disp(' ');
    n_total = n_total + n_images;
    
    %% Data from Broad Institute (BBBC007)
    %  https://data.broadinstitute.org/bbbc/BBBC007/
    %  # of images: 16
    %  # of exluded images: 0
    %  # of network images: 19
    %  # of cells : 1271
    %  Summary: 
    
    disp('Preprocessing BBBC007:      ');
    
    % get information about the quantity of data
    n_images = 0;
    n_cells = 0;
    n_subimages = 0;
    n_excluded = 0;
    
    % setup the image paths
    subfolders = {'A9',...
                  'f96 (17)',...
                  'f113',...
                  'f9620'};
    im_path = fullfile(opts.imgPath,'BBBC007','BBBC007_v1_images');
    lbl_path = fullfile(opts.imgPath,'BBBC007','BBBC007_v1_outlines');
    im_out = fullfile(opts.imgPath,'BBBC007','images');
    lbl_out = fullfile(opts.imgPath,'BBBC007','labels');
    mkdir(im_out);
    mkdir(lbl_out);
    
    % Loop through images
    d = dir(fullfile(im_path,subfolders{1},'*.tif'));
    l = dir(fullfile(lbl_path,subfolders{1},'*.tif'));
    for i = 2:numel(subfolders)
        d = [d; dir(fullfile(im_path,subfolders{i},'*.tif'))];
        l = [l; dir(fullfile(lbl_path,subfolders{i},'*.tif'))];
    end
        
    for i = 1:2:numel(d)
        
        % if images have been processed, then skip
        if exist(fullfile(im_out,d(i).name),'file') && ...
           exist(fullfile(lbl_out,d(i).name),'file')
            raw_images{end+1} = fullfile(im_out,d(i).name);
            seg_images{end+1} = fullfile(lbl_out,d(i).name);
            fprintf('\b\b\b\b\b\b%05.2f%%', i/numel(d)*100);
            continue;
        end
        
        % load image and save nuclear channel
        I = imread(fullfile(d(i).folder,d(i).name));
        n_images = n_images + 1;
        subimages = floor(size(I,1)/200)*floor(size(I,2)/200);
        if subimages == 0
            subimages = 1;
        end
        n_subimages = n_subimages + subimages;
        imwrite(I(:,:,1),fullfile(im_out,d(i).name),'compression','deflate');
        raw_images{end+1} = fullfile(im_out,d(i).name);
        
        % load image and save nuclear channel
        I = imread(fullfile(l(i).folder,d(i).name));
        I = bwareafilt(~I,[25 Inf]);
        F = imfill(I,'holes');
        S = F & ~I;
        S = bwareafilt(S,[5 Inf]);
        F = S | I;
        S = bwlabel(S);
        numObj = max(S(:));
        n_cells = n_cells + double(numObj);
        while ~isequal(F,S>0)
            for j = 1:numObj
                S(imdilate(S==j,ones(3)) & ~S & F) = j;
            end
        end
        S = separate_nuclei(S);
        imwrite(S>0,fullfile(lbl_out,d(i).name),'compression','deflate');
        seg_images{end+1} = fullfile(lbl_out,d(i).name);
        
        fprintf('\b\b\b\b\b\b%05.2f%%', (i+1)/numel(d)*100);

    end
    fprintf('\b\b\b\b\b\b\b100%%', 100);
    warning('on');
    fprintf('\n');
    disp(['Number of images: ' num2str(n_images)]);
    disp(['Number of excluded images: ' num2str(n_excluded)]);
    disp(['Number of network images: ' num2str(n_subimages)]);
    disp(['Number of cells: ' num2str(n_cells)]);
    disp(' ');
    n_total = n_total + n_images;
    
    %% Data from Broad Institute (BBBC006)
    %  https://data.broadinstitute.org/bbbc/BBBC006/
    %  # of images: 768
    %  # of exluded images: 0
    %  # of network images: 3072
    %  # of cells : 192,803
    %  Summary: 
    
    disp('Preprocessing BBBC006:      ');
    
    % get information about the quantity of data
    n_images = 0;
    n_cells = 0;
    n_subimages = 0;
    n_excluded = 0;
    
    % setup the image paths
    im_path = fullfile(opts.imgPath,'BBBC006');
    lbl_path = fullfile(opts.imgPath,'BBBC006','BBBC006_v1_labels');
    im_out = fullfile(opts.imgPath,'BBBC006','images');
    lbl_out = fullfile(opts.imgPath,'BBBC006','labels');
    mkdir(im_out);
    mkdir(lbl_out);
    
    % Loop through images
    d = dir(fullfile(im_path,'BBBC006_v1_images_z_16','*.tif')); % get image folders
    ind = regexp({d.name},'mcf-z-stacks-03212011_([A-Za-z0-9]+)_s([0-9])_w([0-9])','tokens');
    ind = cell2mat(cellfun(@(x) strcmp(x{1}{3},'1'),ind,'UniformOutput',false));
    d = d(ind);
    l = dir(fullfile(lbl_path,'*.png')); % get image folders
    ind = regexp({l.name},'mcf-z-stacks-03212011_([A-Za-z0-9]+)_s([0-9])','tokens');
    for i = 1:numel(l)
        
        % if images have been processed, then skip
        if exist(fullfile(im_out,[l(i).name(1:end-3) 'tif']),'file') && ...
           exist(fullfile(lbl_out,[l(i).name(1:end-3) 'tif']),'file')
            raw_images{end+1} = fullfile(im_out,[l(i).name(1:end-3) 'tif']);
            seg_images{end+1} = fullfile(lbl_out,[l(i).name(1:end-3) 'tif']);
            fprintf('\b\b\b\b\b\b%05.2f%%', i/numel(d)*100);
            continue;
        end
        
        % load image and save labeled image
        I = imread(fullfile(l(i).folder,l(i).name));
        n_images = n_images + 1;
        subimages = floor(size(I,1)/200)*floor(size(I,2)/200);
        if subimages == 0
            subimages = 1;
        end
        n_subimages = n_subimages + subimages;
        L = separate_nuclei(double(I));
        n_cells = n_cells + double(max(L(:)));
        imwrite(L>0,fullfile(lbl_out,[l(i).name(1:end-3) 'tif']),'compression','deflate');
        seg_images{end+1} = fullfile(lbl_out,[l(i).name(1:end-3) 'tif']);
        
        % load image and save nuclear channel
        im_ind = find(startsWith({d.name},l(i).name(1:end-4))); % don't assume files line up
        I = imread(fullfile(d(im_ind).folder,d(im_ind).name));
        imwrite(I,fullfile(im_out,[l(i).name(1:end-3) 'tif']),'compression','deflate');
        raw_images{end+1} = fullfile(im_out,[l(i).name(1:end-3) 'tif']);
        
        fprintf('\b\b\b\b\b\b%05.2f%%', i/numel(l)*100);

    end
    fprintf('\b\b\b\b\b\b\b100%%');
    warning('on');
    fprintf('\n');
    disp(['Number of images: ' num2str(n_images)]);
    disp(['Number of excluded images: ' num2str(n_excluded)]);
    disp(['Number of network images: ' num2str(n_subimages)]);
    disp(['Number of cells: ' num2str(n_cells)]);
    disp(' ');
    n_total = n_total + n_images;
    
    %% Data from Coelho et al.
    %  "NUCLEAR SEGMENTATION IN MICROSCOPE CELL IMAGES: A HAND-SEGMENTED DATASET AND COMPARISON OF ALGORITHMS" by 
    %  Luis Pedro Coelho, Aabid Shariff, and Robert F. Murphy
    %  in International Symposium on Biomedical Imaging (ISBI) 2009
    %  # of images: 97
    %  # of exluded images: 0
    %  # of network images: 1940
    %  # of cells : 6056
    %  Summary: 
    
    disp('Preprocessing Coelho:      ');
    
    % get information about the quantity of data
    n_images = 0;
    n_cells = 0;
    n_subimages = 0;
    n_excluded = 0;
    
    % setup the image paths
    im_path = fullfile(opts.imgPath,'Coelho','data','images','dna-images');
    lbl_path = fullfile(opts.imgPath,'Coelho','data','images','segmented-lpc');
    im_out = fullfile(opts.imgPath,'Coelho','images');
    lbl_out = fullfile(opts.imgPath,'Coelho','labels');
    mkdir(im_out);
    mkdir(lbl_out);
    
    % Loop through images
    d = dir(fullfile(im_path,'gnf','*.png')); % get image folders
    d = [d; dir(fullfile(im_path,'ic100','*.png'))];
    l = dir(fullfile(lbl_path,'gnf','*.tif')); % get image folders
    l = [l; dir(fullfile(lbl_path,'ic100','*.tif'))];
    for i = 1:numel(d)
        
        if i==numel(d)
            break;
        end
        
        % if images have been processed, then skip
        if exist(fullfile(im_out,[d(i).folder(end-3:end) '-' d(i).name(1:end-3) 'tif']),'file') && ...
           exist(fullfile(lbl_out,[l(i).folder(end-3:end) '-' d(i).name(1:end-3) 'tif']),'file')
            raw_images{end+1} = fullfile(im_out,[d(i).folder(end-3:end) '-' d(i).name(1:end-3) 'tif']);
            seg_images{end+1} = fullfile(lbl_out,[l(i).folder(end-3:end) '-' d(i).name(1:end-3) 'tif']);
            fprintf('\b\b\b\b\b\b%05.2f%%', i/numel(d)*100);
            continue;
        end
        
        if ~exist(fullfile(l(i).folder,[d(i).name(1:end-3) 'tif']))
            d(i) = [];
        end
        
        % load image and save nuclear channel
        I = imread(fullfile(d(i).folder,d(i).name));
        n_images = n_images + 1;
        subimages = floor(size(I,1)/200)*floor(size(I,2)/200);
        if subimages == 0
            subimages = 1;
        end
        n_subimages = n_subimages + subimages;
        imwrite(I(:,:,1),fullfile(im_out,[d(i).folder(end-3:end) '-' d(i).name(1:end-3) 'tif']),'compression','deflate');
        raw_images{end+1} = fullfile(im_out,[d(i).folder(end-3:end) '-' d(i).name(1:end-3) 'tif']);
        
        % load labels and save
        I = imread(fullfile(l(i).folder,[d(i).name(1:end-3) 'tif']));
        I = I(:,:,1)>0;
        % fill holes of border objects
        F = imfill(I,'holes');
        F(:,1) = 1; F(1,:) = 1;
        F = imfill(F,'holes');
        F(:,1) = 0; F(:,end) = 1;
        F = imfill(F,'holes');
        F(1,:) = 0; F(end,:) = 1;
        F = imfill(F,'holes');
        F(:,end) = 0; F(:,1) = 1;
        F = imfill(F,'holes');
        S = F & ~I;
        S = bwareafilt(S,[5 Inf]);
        F = S | I;
        S = bwlabel(S,4);
        numObj = max(S(:));
        n_cells = n_cells + double(numObj);
        F_old = F;
        F_same = false;
        while ~F_same
            F_old = F;
            for j = 1:numObj
                S(imdilate(S==j,ones(3)) & ~S & F) = j;
            end
            F_same = isequal(F>0,F_old>0);
        end
        S = separate_nuclei(S);
        imwrite(S>0,fullfile(lbl_out,[l(i).folder(end-3:end) '-' d(i).name(1:end-3) 'tif']),'compression','deflate');
        seg_images{end+1} = fullfile(lbl_out,[l(i).folder(end-3:end) '-' d(i).name(1:end-3) 'tif']);
        
        fprintf('\b\b\b\b\b\b%05.2f%%', i/numel(d)*100);

    end
    fprintf('\b\b\b\b\b\b\b100%%');
    warning('on');
    fprintf('\n');
    disp(['Number of images: ' num2str(n_images)]);
    disp(['Number of excluded images: ' num2str(n_excluded)]);
    disp(['Number of network images: ' num2str(n_subimages)]);
    disp(['Number of cells: ' num2str(n_cells)]);
    disp(' ');
    n_total = n_total + n_images;
    
    %% Initialize the image database
    imdb = struct();
    
    imdb.images.data = zeros([opts.prep.imgSize numel(opts.prep.channels) n_total],'single'); % images
    imdb.images.label = zeros([opts.prep.lblSize numel(opts.prep.channels) n_total],'single'); % label 
    imdb.images.weight = zeros([opts.prep.lblSize numel(opts.prep.channels) n_total],'single');
    imdb.images.set = []; % 1 if image is for training, 2 if image is for validation
    imdb.images.imgFile = {}; % numerical id for image
    imdb.images.lblFile = {};
    
    %% Load files and format data
    disp('-----------------------------------------------')
    disp(' Loading and Formating')
    disp('-----------------------------------------------')
    n_count = 0;
    for i = 1:length(raw_images)
        % Load raw and segmented images
        disp(['Raw Image ' num2str(i) ': ' raw_images{i}])
        raw_img = single(imread(raw_images{i}));
        disp(['Seg Image ' num2str(i) ': ' seg_images{i}])
        seg_img = single(imread(seg_images{i}));
        if size(raw_img,1)~=size(seg_img,1) || size(raw_img,2)~=size(seg_img,2)
            disp('Raw and segmented images are not the same size. Moving to next image.')
            disp('')
            continue
        end
        % images should have the same name, so check it
        img = strsplit(raw_images{i},'/');
        lbl = strsplit(seg_images{i},'/');
        if ~strcmp(img{end},lbl{end})
            warning('Image names do not match. Check for image alignment.');
        end
        
        disp(' ');
        
        % Create weights based on local pixel intensity
        [raw_img,seg_img,weights] = Weight(raw_img,seg_img,opts);
        
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
        
        seg_pixels = seg_img(seg_indices(:));
        seg_pixels = reshape(seg_pixels,opts.prep.lblSize(1),opts.prep.lblSize(2),1,[]);
        weight_pixels = weights(seg_indices(:));
        weight_pixels = reshape(weight_pixels,opts.prep.lblSize(1),opts.prep.lblSize(2),1,[]);
        
        % get the number of cells per image
        % exclude cells touching the top or left side of the image
        for j = 1:size(seg_pixels,4)
            L = bwlabel(seg_pixels(:,:,:,j)>0);
            N = 1:max(L(:));
            E = intersect(N,unique(L(:,1)));
            E = [E;intersect(N,unique(L(1,:))')];
            for k = 1:numel(E)
                N(N==E(k)) = [];
            end
            imdb.images.count(:,:,:,n_count+j) = numel(N);
        end
        
        % Add images to image database
        imdb.images.data(:,:,:,n_count + (1:size(img_pixels,4))) = img_pixels;
        imdb.images.label(:,:,:,n_count + (1:size(img_pixels,4))) = seg_pixels;
        imdb.images.weight(:,:,:,n_count + (1:size(img_pixels,4))) = weight_pixels;
        imdb.images.imgFile = cat(1,imdb.images.imgFile,repmat(raw_images(i),size(seg_pixels,4),1));
        imdb.images.lblFile = cat(1,imdb.images.lblFile,repmat(seg_images(i),size(seg_pixels,4),1));
        n_count = n_count + size(img_pixels,4);
    end
    
    disp(['Formatted ' num2str(size(imdb.images.data,4)) ' images!']);
    if isempty(imdb.images.data)
        return
    end
        
    % Reset the random number generator
    rng(opts.train.randomSeed);
    setIndex = datasample(unique(imdb.images.imgFile),round(numel(unique(imdb.images.imgFile))*opts.prep.holdout));
    imdb.images.set = single(ones(size(imdb.images.label,4),1));
    for i = 1:numel(setIndex)
        imdb.images.set(strcmp(imdb.images.imgFile,setIndex{i})) = 2;
    end

end

function L = separate_nuclei(L)
    S = imboxfilt(double(L>0),3,'NormalizationFactor',1);
    O = imboxfilt(L,3,'NormalizationFactor',1);
    A = O./S;
    L(A ~= L) = 0;
end