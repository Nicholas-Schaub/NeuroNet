function WriteTiff(S,image_path,add_tags)

    t = Tiff(image_path,'w');
    
    switch(class(S))
        case 'logical'
            S = uint8(S);
            sFormat = Tiff.SampleFormat.UInt;
            bps = 8;
        case 'uint8'
            sFormat = Tiff.SampleFormat.UInt;
            bps = 8;
        case 'uint16'
            sFormat = Tiff.SampleFormat.UInt;
            bps = 16;
        case 'uint32'
            sFormat = Tiff.SampleFormat.UInt;
            bps = 32;
        case 'uint64'
            sFormat = Tiff.SampleFormat.UInt;
            bps = 64;
        case 'int8'
            sFormat = Tiff.SampleFormat.Int;
            bps = 8;
        case 'int16'
            sFormat = Tiff.SampleFormat.Int;
            bps = 16;
        case 'int32'
            sFormat = Tiff.SampleFormat.Int;
            bps = 32;
        case 'int64'
            sFormat = Tiff.SampleFormat.Int;
            bps = 64;
        case 'single'
            sFormat = Tiff.SampleFormat.IEEEFP;
            bps = 32;
        case 'double'
            sFormat = Tiff.SampleFormat.IEEEFP;
            bps = 64;
    end
    
    % Setup the tiff file for export
    if numel(S)<2^32-1
        t = Tiff(image_path,'w');
    else
        t = Tiff(image_path,'w8');
    end
    tags.Photometric = Tiff.Photometric.MinIsBlack;
    tags.Compression = Tiff.Compression.AdobeDeflate;
    tags.BitsPerSample = bps;
    tags.SamplesPerPixel = size(S,3);
    tags.SampleFormat = sFormat;
    tags.ImageLength = size(S,1);
    tags.ImageWidth = size(S,2);
    tags.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
    
    if nargin>2
        fields = fieldnames(add_tags);
        for j = fields'
            tags.(j{1}) = add_tags.(j{1});
        end
    end
    
    for j=1:size(S,4)
        t.setTag(tags);
        t.write(S(:,:,:,j));
        if j~=size(S,4)
            t.writeDirectory();
        end
    end
    
    t.close();
end