classdef copyOutput < nnet.layer.RegressionLayer

    methods
        function this = copyOutput(name)
            this.Name = name;
        end

        function loss = forwardLoss(this,Y,T)
            loss = Y;
        end

        function dLdY = backwardLoss(this, Y, T)
            dLdY = Y;
        end
    end
end