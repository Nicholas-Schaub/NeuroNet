function [task,TaskName] = ActivateNextTask(ProcessName)

    pp = mfilename('fullpath');
    pa = strrep(pp,mfilename,'active');
    pp = strrep(pp,mfilename,'tasks');
    
    LockQueue(ProcessName);
    
    q = dir(fullfile(pp,'*.mat'));
    if isempty(q)
        task = [];
        TaskName = [];
        return;
    elseif numel(q)==1
        vals = [str2num(strrep(q.name,'.mat',''))];
    else
        queue = cellfun(@(x) str2num(strrep(x,'.mat','')),{q.name},'UniformOutput',false);
        vals = cell2mat(queue);
    end
    [~,ind] = sort(vals);
    q = q(ind);
    
    OldTaskName = fullfile(q(1).folder,q(1).name);
    task = load(OldTaskName);
    variables = fieldnames(task);
    for variable = 1:numel(variables)
        eval([variables{variable} ' = task.' variables{variable} ';']);
    end
    
    a = dir(fullfile(pa,'*.mat'));
    if any(strcmp(q(1).name,{a.name}))
        if numel(q)==1
            vals = [str2num(strrep(q.name,'.mat',''))];
        else
            queue = cellfun(@(x) str2num(strrep(x,'.mat','')),{q.name},'UniformOutput',false);
            vals = cell2mat(queue);
        end
        TaskName = fullfile(pa,[num2str(max(vals)+1) '.mat']);
    else
        TaskName = fullfile(pa,q(1).name);
    end
    
    taskStart = now;
    lastUpdate = taskStart;
    variables{end+1} = 'taskStart';
    variables{end+1} = 'lastUpdate';
    save(TaskName,variables{:});
    delete(OldTaskName);
    
    UnlockQueue(ProcessName);
    
end