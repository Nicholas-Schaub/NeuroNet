classdef AttributeProto_AttributeType < int32
    %       enum AttributeType {
    %     UNDEFINED = 0;
    %     FLOAT = 1;
    %     INT = 2;
    %     STRING = 3;
    %     TENSOR = 4;
    %     GRAPH = 5;
    %
    %     FLOATS = 6;
    %     INTS = 7;
    %     STRINGS = 8;
    %     TENSORS = 9;
    %     GRAPHS = 10;
    %   }
    enumeration
        UNDEFINED   (0)
        FLOAT       (1)
        INT         (2)
        STRING      (3)
        TENSOR      (4)
        GRAPH       (5)
        FLOATS      (6)
        INTS        (7)
        STRINGS     (8)
        TENSORS     (9)
        GRAPHS      (10)
    end
end
