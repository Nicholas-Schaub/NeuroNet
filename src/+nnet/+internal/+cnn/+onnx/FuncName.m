classdef FuncName < int32
    enumeration
        % These fields must exactly match those of onnxmex.cpp/FuncName, in
        % order, and the numeric values here must be 0,1,...
        
        % The (de)serialize functions either save or load a
        % ModelProtoObject to/from file.
        EdeserializeFromFile        (0)
        EserializeToFile            (1)
        
        % The 'decode' functions take a pointer to an ONNX protobuf object
        % and create a MATLAB object that represents it.
        EdecodeModelProto             (2)
        EdecodeOperatorSetIdProto     (3)
        EdecodeGraphProto             (4)
        EdecodeNodeProto              (5)
        EdecodeAttributeProto         (6)
        EdecodeTensorProto            (7)
        EdecodeTensorProto_Segment    (8)
        EdecodeValueInfoProto         (9)
        EdecodeTypeProto              (10)
        EdecodeTypeProto_Map          (11)
        EdecodeTypeProto_Sequence     (12)
        EdecodeTypeProto_Tensor       (13)
        EdecodeStringStringEntryProto (14)
        EdecodeTensorShapeProto       (15)
        EdecodeTensorShapeProto_Dimension (16)
        
        % The 'encode' functions take a MATLAB object and create a C++ ONNX
        % protobuf object out of it, returning a pointer back to MATLAB.
        EnewModelProto                  (17)
        EencodeModelProto             	(18)
        EencodeOperatorSetIdProto       (19)
        EencodeGraphProto           	(20)
        EencodeNodeProto                (21)
        EencodeAttributeProto        	(22)
        EencodeTensorProto              (23)
        EencodeTensorProto_Segment      (24) 
        EencodeValueInfoProto           (25)
        EencodeTypeProto                (26)
        EencodeTypeProto_Map            (27)
        EencodeTypeProto_Sequence       (28)
        EencodeTypeProto_Tensor         (29)
        EencodeStringStringEntryProto   (30)
        EencodeTensorShapeProto         (31)
        EencodeTensorShapeProto_Dimension (32)
        
        EdestroyModelProto              (33)
    end
end