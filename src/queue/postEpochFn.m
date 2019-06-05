function lr = postEpochFn(net,params,state,TaskName)

    lastUpdate = now;
    save(TaskName,'lastUpdate','-append');
    
    lr = double(state.stats.opt.top1err>0.001)*params.learningRate;
    
end