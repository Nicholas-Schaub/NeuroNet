function [m,s,a]=ReadTiff(filename,startSlice,numSlices)
% [m,s,a]=ReadTiffStack(filename,startSlice,numSlices)
% Returns the stack of images m.  The structure s is returned regardless of
% the number of slices requested.  The pixelsize s.pixA is read from the
% imageDescription field of files that we write with WriteTiffStack;
% otherwise it's returned as zero.  The returned array of structs s are the
% image info returned by iminfo().

if nargin<2
    startSlice=1;
end;
if nargin<3
    numSlices=inf;
end;
a=imfinfo(filename);
nz=numel(a);  % number of slices
start=min(nz,startSlice);
num=max(0,min(nz-startSlice+1,numSlices));
s=struct;

s.nx=a(1).Height;
s.ny=a(1).Width;
s.nz=nz;
s.nc=a(1).BitDepth./a(1).BitsPerSample(1);

% Fill in fields for compatibility with ReadMRC.
s.mx=s.nx;
s.my=s.ny;
s.mz=s.nz;
s.mc=s.nc;

if a(1).BitDepth>8
    m=zeros(s.nx,s.ny,s.nc,num,'uint16');
else
    m=zeros(s.nx,s.ny,s.nc,num,'uint8');
end;
for i=1:num
    m(:,:,:,i)=imread(filename,'tiff','index',i+start-1,'info',a);
end;
