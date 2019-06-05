function UnlockQueue(ProcessName)

    pp = mfilename('fullpath');
    pp = strrep(pp,mfilename,'isLocked.mat');
    
    if ~exist(pp,'file')
        error([ProcessName ': Queue is not currently locked.']);
    end
    
    p = load(pp);
    if ~strcmp(ProcessName,p.ProcessName)
        error([ProcessName ': Queue is currently locked by process ' p.ProcessName '.']);
    end
        
    delete(pp);

end