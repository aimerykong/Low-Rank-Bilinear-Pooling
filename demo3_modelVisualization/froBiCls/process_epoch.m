% -------------------------------------------------------------------------
function  [net_cpu,stats,prof] = process_epoch(net, state, opts, mode)
% -------------------------------------------------------------------------

% initialize empty momentum
if strcmp(mode,'train')
    state.momentum = {} ;
    for i = 1:numel(net.layers)
        if isfield(net.layers{i}, 'weights')
            for j = 1:numel(net.layers{i}.weights)
                state.layers{i}.momentum{j} = 0 ;
            end
        end
    end
end

% move CNN  to GPU as needed
numGpus = numel(opts.gpus) ;
if numGpus >= 1
    net = vl_simplenn_move(net, 'gpu') ;
end
if numGpus > 1
    mmap = map_gradients(opts.memoryMapFile, net, numGpus) ;
else
    mmap = [] ;
end

% profile
if opts.profile
    if numGpus <= 1
        profile clear ;
        profile on ;
    else
        mpiprofile reset ;
        mpiprofile on ;
    end
end

subset = state.(mode) ;
num = 0 ;
stats.num = 0 ; % return something even if subset = []
stats.time = 0 ;
adjustTime = 0 ;
res = [] ;
error = [] ;

start = tic ;
for t=1:opts.batchSize:numel(subset)
    fprintf('%s: epoch %02d: %3d/%3d:', mode, state.epoch, ...
        fix((t-1)/opts.batchSize)+1, ceil(numel(subset)/opts.batchSize)) ;
    batchSize = min(opts.batchSize, numel(subset) - t + 1) ;
    
    for s=1:opts.numSubBatches
        % get this image batch and prefetch the next
        batchStart = t + (labindex-1) + (s-1) * numlabs ;
        batchEnd = min(t+opts.batchSize-1, numel(subset)) ;
        batch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
        num = num + numel(batch) ;
        if numel(batch) == 0, continue ; end
        
        [im, labels] = state.getBatch(state.imdb, batch) ;
        
        if opts.prefetch
            if s == opts.numSubBatches
                batchStart = t + (labindex-1) + opts.batchSize ;
                batchEnd = min(t+2*opts.batchSize-1, numel(subset)) ;
            else
                batchStart = batchStart + numlabs ;
            end
            nextBatch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
            state.getBatch(state.imdb, nextBatch) ;
        end
        
        if numGpus >= 1
            im = gpuArray(im) ;
        end
        
        if strcmp(mode, 'train')
            dzdy = 1 ;
            evalMode = 'normal' ;
        else
            dzdy = [] ;
            evalMode = 'test' ;
        end
        net.layers{end}.class = labels ;
%         res = vl_simplenn(net, im, dzdy, res, ...
%             'accumulate', s ~= 1, ...
%             'mode', evalMode, ...
%             'conserveMemory', opts.conserveMemory, ...
%             'backPropDepth', opts.backPropDepth, ...
%             'sync', opts.sync, ...
%             'cudnn', opts.cudnn) ;
        res = my_simplenn(net, im, dzdy, res, ...
            'accumulate', s ~= 1, ...
            'mode', evalMode, ...
            'conserveMemory', opts.conserveMemory, ...
            'backPropDepth', opts.backPropDepth, ...
            'sync', opts.sync, ...
            'cudnn', opts.cudnn) ;
        
        %{ 
            res(34) % obj value
            %% loss
            res(33) 
            net.layers{33}
            %% bi-layer
            res(32) 
            net.layers{32}
            bilinearDzdx = res(32).dzdx;
            %% scaled-sqrt normalization
            res(31) 
            net.layers{31}
            ssqrtDzdx = res(31).dzdx;
            %% relu5_3
            res(30) 
            net.layers{30}
        %}
        
        % accumulate errors
        error = sum([error, [...
            sum(double(gather(res(end).x))) ;
            reshape(opts.errorFunction(opts, labels, res),[],1) ; ]],2) ;
    end
    
    % accumulate gradient
    if strcmp(mode, 'train')
        if ~isempty(mmap)
            write_gradients(mmap, net) ;
            labBarrier() ;
        end
        [state, net] = accumulate_gradients(state, net, res, opts, batchSize, mmap) ;
    end
    
    % get statistics
    time = toc(start) + adjustTime ;
    batchTime = time - stats.time ;
    stats = extractStats(net, opts, error / num) ;
    stats.num = num ;
    stats.time = time ;
    currentSpeed = batchSize / batchTime ;
    averageSpeed = (t + batchSize - 1) / time ;
    if t == opts.batchSize + 1
        % compensate for the first iteration, which is an outlier
        adjustTime = 2*batchTime - time ;
        stats.time = time + adjustTime ;
    end
    
    fprintf(' %.1f (%.1f) Hz', averageSpeed, currentSpeed) ;
    for f = setdiff(fieldnames(stats)', {'num', 'time'})
        f = char(f) ;
        fprintf(' %s:', f) ;
        fprintf(' %.3f', stats.(f)) ;
    end
    fprintf('\n') ;
    
    % collect diagnostic statistics
    if strcmp(mode, 'train') && opts.plotDiagnostics
        switchfigure(2) ; clf ;
        diagn = [res.stats] ;
        diagnvar = horzcat(diagn.variation) ;
        barh(diagnvar) ;
        set(gca,'TickLabelInterpreter', 'none', ...
            'YTick', 1:numel(diagnvar), ...
            'YTickLabel',horzcat(diagn.label), ...
            'YDir', 'reverse', ...
            'XScale', 'log', ...
            'XLim', [1e-5 1]) ;
        drawnow ;
    end
end

if ~isempty(mmap)
%     unmap_gradients(mmap) ;
end

if opts.profile
    if numGpus <= 1
        prof = profile('info') ;
        profile off ;
    else
        prof = mpiprofile('info');
        mpiprofile off ;
    end
else
    prof = [] ;
end

net_cpu = vl_simplenn_move(net, 'cpu') ;

