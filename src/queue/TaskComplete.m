function TaskComplete(ProcessName,TaskName)

    if ~exist(TaskName,'file')
        error(['The process (' TaskName ') does not exist.'])
    end

    LockQueue(ProcessName);
    
    delete(TaskName);
    
    UnlockQueue(ProcessName);

end