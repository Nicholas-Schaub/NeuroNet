function LockQueue(ProcessName)

    pp = mfilename('fullpath');
    pp = strrep(pp,mfilename,'isLocked.mat');
    
    e = 0;
    while exist(pp,'file')
        try
            p = load(pp);
            disp([ProcessName ': Queue is currently locked by ' p.ProcessName]);
            pause(2*rand);
        catch
            warning([ProcessName ': Could not find the lock file. Will retry...']);
            pause(2*rand);
            e = e+1;
            if e>=5
                warning([ProcessName ': Could not find the lock file after 5 attempts.']);
            end
        end
    end
    
    save(pp,'ProcessName');

end