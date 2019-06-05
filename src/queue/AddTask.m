function AddTask(f,net,imdbpath,opts,ProcessName)

    pp = mfilename('fullpath');
    pp = strrep(pp,mfilename,'tasks');
    
    if ~exist(pp,'dir')
        mkdir(pp);
    end

    LockQueue(ProcessName);
    
    d = dir(fullfile(pp,'*.mat'));
    if isempty(d)
        nextVal = 1;
    elseif numel(d)==1
        nextVal = str2num(strrep(d.name,'.mat',''))+1;
    else
        queue = cellfun(@(x) str2num(strrep(x,'.mat','')),{d.name},'UniformOutput',false);
        nextVal = max(cell2mat(queue))+1;
    end

    TaskName = fullfile(pp,[num2str(nextVal) '.mat']);
    save(TaskName,'f','net','imdbpath','opts');
    
    UnlockQueue(ProcessName);

end