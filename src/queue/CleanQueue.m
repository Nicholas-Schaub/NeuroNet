function CleanQueue(ProcessName,delta)

    delta = delta/(60*24);

    pp = mfilename('fullpath');
    pa = strrep(pp,mfilename,'active');
    pp = strrep(pp,mfilename,'tasks');
    
    if ~exist(pp,'dir')
        mkdir(pp);
    end
    if ~exist(pa,'dir')
        mkdir(pa);
    end

    LockQueue(ProcessName);
    
    actives = dir(fullfile(pa,'*.mat'));
    returnToQueue = {};
    for active = 1:numel(actives)
        a = load(fullfile(actives(active).folder,actives(active).name));
        if a.lastUpdate<now-delta
            returnToQueue{end+1} = actives(active);
        end
    end
    
    d = dir(fullfile(pp,'*.mat'));
    if isempty(d)
        vals = [];
    elseif numel(d)==1
        vals = [str2num(strrep(d.name,'.mat',''))];
    else
        queue = cellfun(@(x) str2num(strrep(x,'.mat','')),{d.name},'UniformOutput',false);
        vals = cell2mat(queue);
    end
    
    [~,ind] = sort(vals);
    d = d(ind);
    for i = (numel(returnToQueue) + [1:numel(d)])
        ind = i-numel(returnToQueue);
        newpath = fullfile(d(ind).folder,[num2str(i) '.mat']);
        oldpath = fullfile(d(ind).folder,d(ind).name);
        if strcmp(newpath,oldpath)
            continue;
        else
            movefile(oldpath,newpath);
        end
    end
    for i = 1:numel(returnToQueue)
        movefile(fullfile(returnToQueue{i}.folder,returnToQueue{i}.name),fullfile(d(i).folder,[num2str(i) '.mat']));
    end
    
    UnlockQueue(ProcessName);
    
end

