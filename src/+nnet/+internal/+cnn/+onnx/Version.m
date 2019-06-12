classdef Version < int32
    % enum Version {
    %   _START_VERSION = 0;
    %   IR_VERSION_2017_10_10 = 0x00000001;
    %   IR_VERSION_2017_10_30 = 0x00000002;
    %   IR_VERSION = 0x00000003;
    % }
    enumeration
        START_VERSION           (0)
        IR_VERSION_2017_10_10   (1)
        IR_VERSION_2017_10_30   (2)
        IR_VERSION              (3)
    end
end