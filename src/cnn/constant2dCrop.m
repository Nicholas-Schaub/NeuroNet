classdef constant2dCrop < nnet.layer.Layer

    properties
        crop_ref
    end

    properties (Constant)
        DefaultName = 'cropRef';
    end
    
    methods
        function this = constant2dCrop(crop_size,name)
            this.crop_ref = crop_size;
            this.Name = name;
        end
        
        function Z = predict(this,X)
            Z = this.forward(X);
        end

        function [Z,memory] = forward(this,X)
            Z = X((1:this.crop_ref(4))+this.crop_ref(2),(1:this.crop_ref(3))+this.crop_ref(1),:,:);
            memory = 0;
        end

        function dLdY = backward(this, X, Z, dLdZ, memory)
            dLdY = X;
        end
    end
end