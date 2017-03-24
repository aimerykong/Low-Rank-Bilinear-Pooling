function [net, stats] = cnn_trainMyModel(net, imdb, prefixStr, getBatch, varargin)
%CNN_TRAIN  An example implementation of SGD for training CNNs
%    CNN_TRAIN() is an example learner implementing stochastic
%    gradient descent with momentum to train a CNN. It can be used
%    with different datasets and tasks by providing a suitable
%    getBatch function.
%
%    The function automatically restarts after each training epoch by
%    checkpointing.
%
%    The function supports training on CPU or on one or more GPUs
%    (specify the list of GPU IDs in the `gpus` option).


opts.expDir = fullfile('data','exp') ;
opts.continue = true ;
opts.batchSize = 256 ;
opts.numSubBatches = 1 ;
opts.train = [] ;
opts.val = [] ;
opts.gpus = [] ;
opts.prefetch = false ;
opts.numEpochs = 300 ;
opts.learningRate = 0.001 ;
opts.weightDecay = 0.0005 ;
opts.momentum = 0.9 ;
opts.randomSeed = 0 ;
opts.memoryMapFile = fullfile(tempdir, 'matconvnet.bin') ;
opts.profile = false ;

opts.conserveMemory = true ;
opts.backPropDepth = +inf ;
opts.sync = false ;
opts.cudnn = true ;
opts.errorFunction = 'multiclass' ;
opts.errorLabels = {} ;
opts.plotDiagnostics = false ;
opts.plotStatistics = true;
opts = vl_argparse(opts, varargin) ;

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end
if isnan(opts.train), opts.train = [] ; end
if isnan(opts.val), opts.val = [] ; end

% -------------------------------------------------------------------------
%                                                            Initialization
% -------------------------------------------------------------------------

net = vl_simplenn_tidy(net); % fill in some eventually missing values
net.layers{end-1}.precious = 1; % do not remove predictions, used for error
vl_simplenn_display(net, 'batchSize', opts.batchSize) ;

evaluateMode = isempty(opts.train) ;
if ~evaluateMode
    for i=1:numel(net.layers)
        if isfield(net.layers{i}, 'weights')
            J = numel(net.layers{i}.weights) ;
            if ~isfield(net.layers{i}, 'learningRate')
                net.layers{i}.learningRate = ones(1, J, 'single') ;
            end
            if ~isfield(net.layers{i}, 'weightDecay')
                net.layers{i}.weightDecay = ones(1, J, 'single') ;
            end
        end
    end
end

% setup error calculation function
hasError = true ;
if isstr(opts.errorFunction)
    switch opts.errorFunction
        case 'none'
            opts.errorFunction = @error_none ;
            hasError = false ;
        case 'multiclass'
%             opts.errorFunction = @error_multiclass ;
            opts.errorFunction = @my_error_multiclass ;
            if isempty(opts.errorLabels)
                opts.errorLabels = {'top1err', 'top5err'} ; 
            end
        case 'binary'
            opts.errorFunction = @error_binary ;
            if isempty(opts.errorLabels)
                opts.errorLabels = {'binerr'} ; 
            end
        otherwise
            error('Unknown error function ''%s''.', opts.errorFunction) ;
    end
end

state.getBatch = getBatch ;
stats = [] ;

% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------

modelPath = @(ep) fullfile(opts.expDir, sprintf('%snet-epoch-%d.mat', prefixStr, ep));
modelFigPath = fullfile(opts.expDir, [prefixStr 'net-train.pdf']) ;

start = opts.continue * myfindLastCheckpoint(opts.expDir, prefixStr) ;
if start >= 1
    fprintf('%s: resuming by loading epoch %d\n', mfilename, start) ;
    [net, stats] = loadState(modelPath(start)) ;
end

net.layers{30}.precious = 1; %% intervene this layer, comment later
net.layers{31}.precious = 1; %% intervene this layer, comment later

%{
net.layers{25}.learningRate = 1*[1 2]; % conv5_3
net.layers{27}.learningRate = 1*[1 2]; % conv5_3
net.layers{29}.learningRate = 1*[1 2]; % conv5_3
net.layers{31}.learningRate = 0.1*[1 2]; % scaled sqrt
net.layers{32}.learningRate = 0.1*[1 2]; % pseudo-bilinear-layer

net.layers{25}.weightDecay = [0.0005,0]; %ones(1, J, 'single') ;
net.layers{27}.weightDecay = [0.0005,0]; %ones(1, J, 'single') ;
net.layers{29}.weightDecay = [0.0005,0]; %ones(1, J, 'single') ;
net.layers{31}.weightDecay = [0.0005,0]; %ones(1, J, 'single') ;
net.layers{32}.weightDecay = [0.0005,0]; %ones(1, J, 'single') ;
%}
for i = 18:numel(net.layers)
    if isfield(net.layers{i}, 'weights')
        J = numel(net.layers{i}.weights) ;
        if J~=0 && strcmp(net.layers{i}.type,'conv')
            net.layers{i}.learningRate = 0.01*ones(1, J, 'single') ;
        elseif J~=0 && ~strcmp(net.layers{i}.type,'conv')
            net.layers{i}.learningRate = 0.1*[0.1 0.2]; %0*ones(1, J, 'single') ;6
        end
        
        if J~=0 && strcmp(net.layers{i}.type,'conv')
            net.layers{i}.weightDecay = [0.001,0]; %ones(1, J, 'single') ;
        elseif J~=0 && ~strcmp(net.layers{i}.type,'conv')
            net.layers{i}.weightDecay = [0.001,0]; %ones(1, J, 'single') ;
        end
    end
end
net.layers{34}.learningRate = [1 2]*0.1 ;


for epoch = start+1:opts.numEpochs
    
    % Set the random seed based on the epoch and opts.randomSeed.
    % This is important for reproducibility, including when training
    % is restarted from a checkpoint.
    
    rng(epoch + opts.randomSeed) ;
    prepareGPUs(opts, epoch == start+1) ;
    
    % Train for one epoch.
    
    state.epoch = epoch ;
    state.learningRate = opts.learningRate(min(epoch, numel(opts.learningRate))) ;
    state.train = opts.train(randperm(numel(opts.train))) ; % shuffle
    state.val = opts.val(randperm(numel(opts.val))) ;
    state.imdb = imdb ;
    
    if numel(opts.gpus) <= 1
        [net,stats.train(epoch),prof] = process_epoch(net, state, opts, 'train') ;
        [~,stats.val(epoch)] = process_epoch(net, state, opts, 'val') ;
        if opts.profile
            profview(0,prof) ;
            keyboard ;
        end
    else
        spmd(numGpus)
            [net_, stats_.train, prof_] = process_epoch(net, state, opts, 'train') ;
            [~, stats_.val] = process_epoch(net_, state, opts, 'val') ;
            if labindex == 1, savedNet_ = net_ ; end
        end
        net = savedNet_{1} ;
        stats__ = accumulateStats(stats_) ;
        stats.train(epoch) = stats__.train ;
        stats.val(epoch) = stats__.val ;
        if opts.profile
            mpiprofile('viewer', [prof_{:,1}]) ;
            keyboard ;
        end
        clear net_ stats_ stats__ savedNet_ ;
    end
    
    % save
    if ~evaluateMode
        saveState(modelPath(epoch), net, stats) ;
    end
    
    if opts.plotStatistics
        switchFigure(1) ; clf ;
        plots = setdiff(...
            cat(2,...
            fieldnames(stats.train)', ...
            fieldnames(stats.val)'), {'num', 'time'}) ;
        for p = plots
            p = char(p) ;
%             values = zeros(0, epoch) ;
            values = zeros(0, length(stats.train)) ;
            leg = {} ;
            for f = {'train', 'val'}
                f = char(f) ;
                if isfield(stats.(f), p)
                    tmp = [stats.(f).(p)] ;
                    values(end+1,:) = tmp(1,:)' ;
                    leg{end+1} = f ;
                end
            end
            subplot(1,numel(plots),find(strcmp(p,plots))) ;
%             plot(1:epoch, values','.-') ; % original
            plot(1:size(values,2), values(:, 1:end)','o-') ; % don't plot the first epoch
%             plot(1:epoch, values(:, 1:end)','o-') ; % don't plot the first epoch
            xlabel('epoch') ;
            [minVal,minIdx] = min(values(2,:));
            title(sprintf('%s tsErr%.4f (%d) trErr%.4f', p, min(values(2,:)), minIdx, min(values(1,:)))) ;
%             title(sprintf('%s tsErr%.4f trErr%.4f', p, min(values(2,:)), min(values(1,:)))) ;
            %title(p);
            legend(leg{:},'location', 'SouthOutside') ;
            grid on ;
        end
        drawnow;
        
        [curpath, curname, curext] = fileparts(modelFigPath);        
%         saveas(gcf, fullfile(curpath, [curname, '.png']) );
        export_fig(fullfile(curpath, [curname, '.png']), '-png');
%         print(1, modelFigPath, '-dpdf') ;
    end
end




% -------------------------------------------------------------------------
% function unmap_gradients(mmap)
% -------------------------------------------------------------------------




