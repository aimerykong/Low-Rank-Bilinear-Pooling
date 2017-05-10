% -------------------------------------------------------------------------
function prepareGPUs(opts, cold)
% -------------------------------------------------------------------------
numGpus = numel(opts.gpus) ;
if numGpus > 1
    % check parallel pool integrity as it could have timed out
    pool = gcp('nocreate') ;
    if ~isempty(pool) && pool.NumWorkers ~= numGpus
        delete(pool) ;
    end
    pool = gcp('nocreate') ;
    if isempty(pool)
        parpool('local', numGpus) ;
        cold = true ;
    end
    if exist(opts.memoryMapFile)
        delete(opts.memoryMapFile) ;
    end
end
if numGpus >= 1 && cold
    fprintf('%s: resetting GPU\n', mfilename)
    if numGpus == 1
        gpuDevice(opts.gpus)
    else
        spmd, gpuDevice(opts.gpus(labindex)), end
    end
end