%% quick setup
clear
% close all
clc;

run(fullfile('./vlfeat','toolbox','vl_setup'));
run(fullfile('./matconvnetToolbox', 'matlab', 'vl_setupnn'));
addpath(genpath('./froBiCls'));
addpath(genpath('./addLayerFunc'));
addpath(genpath('./exportFig'));
% addpath(genpath('./get_activations'));
% addpath(genpath('./get_batch'));
% addpath(genpath('./linear_classifier'));
% addpath(genpath('./prepare_dataset'));
% addpath(genpath('../layers'));
% addpath(genpath(fullfile('../matconvnetToolbox','examples')));
%% configuration
% dataset: 'CUB' (bird), 'MIT' (indoor scene), 'FMD' (fclikr material),
% 'DTD' (describable texture), 'aircraft' (ox), 'cars' (stanford)
dataset = 'CUB';

% network: VGG_M, VGG_16, VGG_19, resnet-152, resnet-50, resnet-101
netbasemodelName = 'VGG_16';

gpuId = 1;
gpuDevice(gpuId);

learningRate = [ones(1, 300)*0.01, ones(1, 200)*0.001, ones(1, 200)*0.0001];
% weightDecay: usually use the default value
weightDecay=0.0005;
%% prepare data
% dataset: 'CUB', 'MIT', 'DTD', 'aircrafts', 'cars'
if strcmp(dataset, 'CUB')
    num_classes = 200;
    dataDir = './data/cub';
%     imdbFile = fullfile('../imdbFolder', dataset, [lower(dataset) '-seed-01'], 'imdb-seed-1.mat');
    imdbFile = fullfile('./imdb-seed-1.mat');
    if ~exist(imdbFile, 'file')
        imdb = cub_get_database(dataDir);
        imdbDir=fileparts(imdbFile);
        if ~isdir(imdbDir)
            mkdir(imdbDir);
        end
        save(imdbFile, '-struct', 'imdb') ;
    end
elseif strcmp(dataset, 'DTD')
    num_classes = 47;
    dataDir = '../data/dtd';
    imdbFile = fullfile('../imdbFolder', dataset, [lower(dataset) '-seed-01'], 'imdb-seed-1.mat');
    if ~exist(imdbFile, 'file')
        imdb = dtd_get_database(dataDir);
        
        imdbDir = fileparts(imdbFile);
        if ~isdir(imdbDir)
            mkdir(imdbDir);
        end
        save(imdbFile, '-struct', 'imdb') ;
    end
end
%% read pre-trained model and initialize network
%%%%[[!! put the model here ---
%%%%  https://drive.google.com/open?id=0BxeylfSgpk1MOGNPZkxiWE1NUUU !!]]
netbasemodel = load('./CUB_VGG_16_SVM_bilinear_448_net-epoch-23.mat') ;
netbasemodel = netbasemodel.net;
netbasemodel.layers = netbasemodel.layers(1:end-1);
netbasemodel = vl_simplenn_tidy(netbasemodel);
netbasemodel = vl_simplenn_move(netbasemodel, 'gpu') ;
vl_simplenn_display(netbasemodel);
%% testing
% imdbFile = fullfile('../imdbFolder', dataset, [lower(dataset) '-seed-01'], 'imdb-seed-1.mat');
imdbFile = fullfile('./imdb-seed-1.mat');
imdb = load(imdbFile) ;
nclass = numel(imdb.meta.classes);
mode = 'test';
tsIdx = find(imdb.images.set==3);
% tsIdx = tsIdx(end-9:end);

% tsIdx = find(imdb.images.set~=1);
scoreList = zeros(nclass, length(tsIdx), 'single');
scoreMaxList = zeros(nclass, length(tsIdx), 'single');
grndLabel = zeros(1,length(tsIdx), 'single');
optsImg.imageSize = [448, 448] ;
optsImg.border = [29, 29] ;
optsImg.keepAspect = true ;
optsImg.averageImage = netbasemodel.meta.normalization.averageImage;
optsImg.numAugments = 1;
%% generate the ground-truth label in order to backpropagate expected activation
netbasemodelFeedforward = netbasemodel;

lambda = 0;
netbasemodel.layers = netbasemodel.layers(1:end-1);
rDim = netbasemodelFeedforward.layers{37}.rDim;
W = netbasemodelFeedforward.layers{37}.weights{1};
b = netbasemodelFeedforward.layers{37}.weights{2};
netbasemodel = addFroNormAct(netbasemodel, {{W,b}}, lambda, rDim, nclass);
netbasemodel.layers{end+1} = struct('type', 'loss',...
    'name', 'final_loss', ...
    'class', 'euclidean', ...
    'precious', 1);

vl_simplenn_display(netbasemodel);

for imgIdx = 1:10% length(tsIdx) % 220
    %% feed an image
    tsImgID = tsIdx(imgIdx); % end
    label = imdb.images.label(tsImgID) ;
%     imageName = strcat('../data/cub/images/', imdb.images.name(tsImgID)) ;
    imageName = strcat('./cub/images/', imdb.images.name(tsImgID)) ;
    [imo, imBackup] = fetchCUB4BackpropAct(imageName, mode, optsImg) ;
    
    netbasemodelFeedforward.layers{33}.precious=1;
    res = my_simplenn(netbasemodelFeedforward, gpuArray(imo), [], [], ...
        'accumulate', 0, ...
        'mode', mode, ...
        'conserveMemory', 1, ...
        'cudnn', true) ;
    
    relu5_3 = gather(res(34).x); 
    res37 = gather(res(37).x);
    relu5_3Mean = mean(relu5_3,3);
    res37Mean = mean(res37,3);
    res37AbsMean = mean(abs(res37),3);
        
    rowNum = 2;
    columnNum = 5;
    
    
    figure(1);clf(1);
    subplot(rowNum,columnNum,1); subwindow = 1; subwindow = subwindow + 1;
    imshow(uint8(imBackup)); axis off image; title('orgImg'); % colorbar;
    subplot(rowNum,columnNum,subwindow); subwindow = subwindow + 1;
    imagesc(relu5_3Mean); axis off image; title('mean relu53'); % colorbar;
    subplot(rowNum,columnNum,subwindow); subwindow = subwindow + 1;
    imagesc(res37Mean); axis off image; title('mean bicls'); % colorbar;
    subplot(rowNum,columnNum,subwindow); subwindow = subwindow + 1;
    imagesc(res37AbsMean); axis off image; title('abs mean bicls'); % colorbar;
    
    rDim = netbasemodel.layers{37}.rDim;
    W = netbasemodel.layers{37}.weights{1};
    b = netbasemodel.layers{37}.weights{2};
    froTensor = reshape(res37, [28*28, 100]);
    froTensor = W' * froTensor';
    froTensor = reshape(froTensor', [28, 28, rDim, nclass] );
    
    Ypred = froTensor;
    Ygt = froTensor;
    amplifier = 1.0;
    for i = 1:nclass
        if i == label
            Ygt(:,:,1:rDim/2,i) = amplifier*Ygt(:,:,1:rDim/2,i); % positive
            Ygt(:,:,rDim/2+1:end,i) = 0; % negative
        else
            Ygt(:,:,1:rDim/2,i) = 0; % positive
            Ygt(:,:,rDim/2+1:end,i) = amplifier*Ygt(:,:,rDim/2+1:end,i); % negative
        end
    end
    % norm(Ypred(:) - Ygt(:))
    %% setup to backpropagate
    Ygt = reshape(Ygt, [28*28, rDim*nclass]);
    Ygt = Ygt';
    
    netbasemodel.layers{end}.class = Ygt ;
    res = backpropAct_simplenn(netbasemodel, gpuArray(imo), [], [], ...
        'accumulate', 0, ...
        'mode', mode, ...
        'conserveMemory', 1, ...
        'cudnn', true) ;
    
    for i = 1:numel(netbasemodel.layers)
        if isfield(netbasemodel.layers{i}, 'weights')
            J = numel(netbasemodel.layers{i}.weights) ;
            if J~=0 && strcmp(netbasemodel.layers{i}.type,'conv')
                netbasemodel.layers{i}.learningRate =   0.00*ones(1, J, 'single') ;
            elseif J~=0 && strcmp(netbasemodel.layers{i}.name,'learnablePowerNormLayer4biCls')
                % learnable power normalization layer
                netbasemodel.layers{i}.learningRate = 0;
            elseif J~=0 && strcmp(netbasemodel.layers{i}.type,'bnorm')
                netbasemodel.layers{i}.learningRate = 0*  0.01*[1 1 0.05]; % default
            elseif J~=0 && strcmp(netbasemodel.layers{i}.name,'biCls') % the bilinear SVM layer
                netbasemodel.layers{i}.learningRate = 0*  0.002*[1 2];
            end
            
            if J~=0 && strcmp(netbasemodel.layers{i}.type,'conv')
                netbasemodel.layers{i}.weightDecay = 0*[0.01,0]; %ones(1, J, 'single') ;
            elseif J~=0 && strcmp(netbasemodel.layers{i}.name,'learnablePowerNormLayer4biCls')
                % learnable power normalization layer
                netbasemodel.layers{i}.weightDecay = 0;
            elseif J~=0 && strcmp(netbasemodel.layers{i}.type,'bnorm')
                netbasemodel.layers{i}.weightDecay = 0*netbasemodel.layers{i}.weightDecay; % default
            elseif J~=0 && strcmp(netbasemodel.layers{i}.name,'biCls') % the bilinear SVM layer
                netbasemodel.layers{i}.weightDecay = 0*[0.01 0];
                netbasemodel.layers{i}.lambda = 1;
            end
        end
    end
    
    netbasemodel.layers{end}.class = Ygt ;
    netbasemodel = vl_simplenn_tidy(netbasemodel);
    netbasemodel = vl_simplenn_move(netbasemodel, 'gpu') ;
    
    dzdy = 1;
    s = 1;
    res = [] ;
    evalMode = 'normal';
    
    im = gpuArray(imo);
    res = backpropAct_simplenn(netbasemodel, im, dzdy, res, ...
        'accumulate', s~=1, ...
        'mode', evalMode, ...
        'conserveMemory', true, ...
        'backPropDepth', inf, ...
        'sync', false, ...
        'cudnn', true) ;
    %% tweaking the input image
    lr = 10e11;
    MAXITER = 200;
    lossList = zeros(1,MAXITER);
    imBackup = gpuArray(imo);
    im = gpuArray(imo);
    lambda = 0;%10^(-14);
    for iter = 1:MAXITER
        lossList(iter) = gather(res(end).x);
        %     im = im - lr*(res(1).dzdx );
        im = im - lr*(res(1).dzdx + (im-imBackup)*lambda);
        
        %     momentum = 0.9 - 0.001* im - res(1).dzdx;
        %     im = im + lr * momentum;
        
        res = backpropAct_simplenn(netbasemodel, im, dzdy, res, ...
            'accumulate', s~=1, ...
            'mode', evalMode, ...
            'conserveMemory', true, ...
            'backPropDepth', inf, ...
            'sync', false, ...
            'cudnn', true) ;
        fprintf('iter-%04d, loss=%.9f\n', iter, lossList(iter));
    end
    
    % figure(2);
    % subplot(rowNum,columnNum,1);
    % imInput = bsxfun(@plus, imo, netbasemodel.meta.normalization.averageImage);
    % imshow(uint8(imInput)); title('original input image');
    
    %% show figure    
    imTweak = bsxfun(@plus, im, netbasemodel.meta.normalization.averageImage);
    
    imTweakSqrt = imTweak;
    signTMP = sign(imTweakSqrt);
    imTweakSqrt = abs(imTweakSqrt).^0.5;
    imTweakSqrt = signTMP.*imTweakSqrt;
    imTweakSqrt = imTweakSqrt - min(imTweakSqrt(:));
    imTweakSqrt = imTweakSqrt ./ max(imTweakSqrt(:));
    
    imTweak = imTweak - min(imTweak(:));
    imTweak = imTweak / max(imTweak(:));
    
    subplot(rowNum,columnNum,subwindow); subwindow = subwindow + 1;
    imshow((imTweakSqrt)); title('tweaked-sqrt');    
    
    subplot(rowNum,columnNum,subwindow); subwindow = subwindow + 1;
    imTweakSqrtGauss = imgaussfilt(imTweakSqrt, 1, 'FilterSize', [5,5]);
    imshow((imTweak)); title('tweaked-sqrt-gauss');
    
    
    subplot(rowNum,columnNum,subwindow); subwindow = subwindow + 1;
    imshow((imTweak)); title('tweaked input image');
        
    subplot(rowNum,columnNum,subwindow); subwindow = subwindow + 1;
    imDiff = imo-im;
    imDiffAbs = abs(imDiff);
    imDiff = imDiff - min(imDiff(:));
    imDiff = imDiff ./ max(imDiff(:));
    imshow(imDiff); title('diff');
    
    subplot(rowNum,columnNum,subwindow); subwindow = subwindow + 1;
    imDiffAbs = imDiffAbs - min(imDiffAbs(:));
    imDiffAbs = imDiffAbs ./ max(imDiffAbs(:));
    imshow(imDiffAbs); title('abs diff');
    
    subplot(rowNum,columnNum,subwindow); subwindow = subwindow + 1;
    plot(1:length(lossList), lossList, 'r-'); title('diff curve');
    %% output to screen and save figure
    fprintf('\nclass-%d\n', label);
    export_fig(sprintf('testCls%d_Img%d_2.fig', label, imgIdx));
end