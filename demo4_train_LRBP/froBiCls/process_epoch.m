function  [net_cpu,stats,prof] = process_epoch(net, state, opts, mode)
%% initialize empty momentum
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

% num_classes = 200;
% load('caffe_ExtmAugWithVal.mat');
% W=W(:,1:num_classes);
% b=b(1:num_classes);
% for i = 1:num_classes
%     ww = W(:,i);
%     ww = reshape(ww,[512 512]);
%     %ww = ww';
%     ww = (ww+ww')/2;
%     W(:,i) = ww(:);
% end
% W = single(reshape(W, [1,1,512^2,200]));
% b = single(b(:));
% net.layers{37}.weights = {single(W),single(b)};
% % netW = gather(net.layers{37}.weights{1});
% net.normalization.imageSize=double([448 448 3]);
% net.normalization.averageImage = net.meta.normalization.averageImage;


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
% opts.batchSize = 1;
% valFV = [];
% score = [];
% grndLabel = [];
% net.layers{36}.precious = 1;

%{
num_classes = 200;
% net.layers = net.layers(1:36);
net2=net;
net2.layers = net2.layers(1:37)
batchSize=1;
mopts.poolType='bilinear';
mopts.use448 = true;
mopts.classifyType='SVM'; % or SVM or LR
mopts.initMethod='FroBiSVM'; % or 'random' 'FroBiSVM' 'symmetricFullSVM'
[valFV, valY] = get_activations_dataset_network_main014testOnly(...
    'CUB', 'VGG_16', mopts.poolType, mopts.classifyType, mopts.initMethod, mopts.use448, net2, batchSize);
% load('caffe_ExtmAugWithVal.mat');
% W=W(:,1:num_classes);
% b=b(1:num_classes);
% for i = 1:num_classes
%     ww = W(:,i);
%     ww = reshape(ww,[512 512]);
%     %ww = ww';
%     ww = (ww+ww')/2;
%     W(:,i) = ww(:);
% end
% score = bsxfun(@plus, W'*valFV, b(:));
score = valFV;
[~,predLabel] = max(score,[],1);
grndLabel = valY(:);
acc = mean(grndLabel(:)==predLabel(:))
%}


% net = load('./imdbFolder/CUB/exp/CUB_VGG_16_SVM_bilinear_448_main012_20rank_init020learnablePower_bicls_BN3/version10/CUB_VGG_16_SVM_bilinear_448_net-epoch-17.mat') ;
% net = net.net;
% net.layers = net.layers(1:34);
% alpha = 0;
% ep = 0.001;
% net.layers = net.layers(1:34);
% net = addAveMaxAdjustLayer(net, alpha, ep);
% net = addBilinear(net);
% num_classes = 200;
% caffeTrainedSVM = load('caffe_ExtmAugWithVal.mat');
% W = caffeTrainedSVM.W(:,1:num_classes);
% b = caffeTrainedSVM.b(1:num_classes);
% for ii = 1:num_classes
%     ww = W(:,ii);
%     ww = reshape(ww,[512 512]);
%     %ww = ww';
%     ww = (ww+ww')/2;
%     W(:,ii) = ww(:);
% end
% W = single(reshape(W, [1,1,512^2,200]));
% b = single(b(:));
% initFCparam = {{W,b}};
% net = addFC(net, 'fc_final', initFCparam, 'specified');
% net = vl_simplenn_tidy(net) ;
% net.normalization.imageSize=double([448 448 3]);
% net.normalization.averageImage = net.meta.normalization.averageImage;
% net=vl_simplenn_move(net, 'gpu');
% net.layers{end+1}=struct('type', 'loss',...
%                              'name', 'final_loss', ...
%                              'lossType', 'hinge');


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
        if numel(batch) == 0, 
            continue ; 
        end
        
%         [im, labels] = state.getBatch(state.imdb, batch,'test');
        [im, labels] = state.getBatch(state.imdb, batch, mode) ;
%         [im, labels] = state.getBatch(state.imdb, batch, mode, ...
%             'rgbVariance', net.meta.normalization.rgbVariance) ;

        if opts.prefetch
            if s == opts.numSubBatches
                batchStart = t + (labindex-1) + opts.batchSize ;
                batchEnd = min(t+2*opts.batchSize-1, numel(subset)) ;
            else
                batchStart = batchStart + numlabs ;
            end
            nextBatch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
%             state.getBatch(state.imdb, nextBatch, 'test') ;
            state.getBatch(state.imdb, nextBatch, mode) ;
%             [im, labels] = state.getBatch(state.imdb, batch, mode, ...
%                 'rgbVariance', net.meta.normalization.rgbVariance) ;
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
%             evalMode = 'val' ;
        end
        net.layers{end}.class = labels ;
        %{
        
%         net.layers{24}.precious = 1;
%         net.layers{25}.precious = 1;
%         net.layers{26}.precious = 1;
%         net.layers{27}.precious = 1;
%         net.layers{28}.precious = 1;
%         res = vl_simplenn(net, im, [], [], 'conserveMemory', true);
        
%         A25=squeeze(gather(res(25).x));
%         A26=squeeze(gather(res(26).x));
%         A27=squeeze(gather(res(27).x));
%         A28=squeeze(gather(res(28).x));
%         A29=squeeze(gather(res(29).x));
%         A37=squeeze(gather(res(37).x)); 
%         A38=squeeze(gather(res(38).x));
        %}
        net.layers{end}.class = labels ;   
        res = my_simplenn(net, im, dzdy, res, ...
            'accumulate', s~=1, ...
            'mode', evalMode, ...
            'conserveMemory', opts.conserveMemory, ...
            'backPropDepth', opts.backPropDepth, ...
            'sync', opts.sync, ...
            'cudnn', opts.cudnn) ;
        %{
%         B25=squeeze(gather(res(25).x));
%         B26=squeeze(gather(res(26).x));
%         B27=squeeze(gather(res(27).x));
%         B28=squeeze(gather(res(28).x));
%         B29=squeeze(gather(res(29).x));
%         B37=squeeze(gather(res(37).x)); 
%         B38=squeeze(gather(res(38).x));
       %}
            
        
%         [curImages, labels]=state.getBatch(state.imdb, batch,'test');
%         getBatchFunc(imdb, batchIds(getInter(i+1, batchSize, numel(batchIds))),'test');     
%         net.layers{end}.class = labels ;   
%         res = vl_simplenn(net, gpuArray(curImages),[],[], 'conserveMemory', true); %acc=0.818778
        
        
        
%         size(gather(squeeze(res(37).x)))
%         valFV = [valFV, gather(squeeze(res(37).x))];
%         score = [score, gather(squeeze(res(end-1).x))];
%         grndLabel = [grndLabel; gather(labels(:))];
        
        % accumulate errors
        error = sum(...
            [error, ...
            [sum(double(gather(res(end).x))) ;
            reshape(opts.errorFunction(opts, labels, res),[],1) ; ] ]...
            , 2) ;
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

% [~,predLabel] = max(score,[],1);
% acc = mean(grndLabel(:)==predLabel(:))

% load('caffe_ExtmAugWithVal.mat');
% score2 = bsxfun(@plus, W'*valFV, b(:));
% [~,predLabel] = max(score2,[],1);
% acc = mean(grndLabel(:)==predLabel(:))

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

