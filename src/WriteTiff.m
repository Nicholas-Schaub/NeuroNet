function WriteTiff(S,image_path)

    t = Tiff(image_path,'w');
    
    switch(class(S))
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
    t.setTag('Photometric',Tiff.Photometric.MinIsBlack);
    t.setTag('Compression',Tiff.Compression.AdobeDeflate);
    t.setTag('BitsPerSample',bps);
    t.setTag('SamplesPerPixel',size(S,3));
    t.setTag('SampleFormat',sFormat);
    t.setTag('ImageLength',size(S,1));
    t.setTag('ImageWidth',size(S,2));
    t.setTag('PlanarConfiguration',Tiff.PlanarConfiguration.Chunky);

    t.write(S);
    t.close();
end