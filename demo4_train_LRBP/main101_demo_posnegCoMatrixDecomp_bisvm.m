% quick setup
clear
% close all
clc;

% run(fullfile('./vlfeat','toolbox','vl_setup'));
addpath(genpath('froBiCls'));
addpath(genpath('addLayerFunc'));
addpath(genpath('get_activations'));
addpath(genpath('get_batch'));
addpath(genpath('linear_classifier'));
addpath(genpath('prepare_dataset'));
addpath(genpath('layers'));
run(fullfile('matconvnetToolbox', 'matlab', 'vl_setupnn'));
addpath(genpath(fullfile('matconvnetToolbox','examples')));
addpath(genpath('./exportFig'));
%% configuration
% dataset: 'CUB' (bird), 'MIT' (indoor scene), 'FMD' (fclikr material), 
% 'DTD' (describable texture), 'aircraft' (ox), 'cars' (stanford)
dataset = 'CUB';

% network: VGG_M, VGG_16, VGG_19, resnet-152, resnet-50, resnet-101
netbasemodelName = 'VGG_16';

gpuId = 1;
gpuDevice(gpuId);

learningRate = [ones(1, 100)*0.01 ones(1, 50)*0.01 ones(1, 50)*0.0001];
% weightDecay: usually use the default value
weightDecay=0.0005;
%% prepare data
% dataset: 'CUB', 'MIT', 'DTD', 'aircrafts', 'cars'
if strcmp(dataset, 'CUB')
    num_classes = 200;
    dataDir = './data/cub';
    imdbFile = fullfile('imdbFolder', dataset, [lower(dataset) '-seed-01'], 'imdb-seed-1.mat');
%     imdbFile = fullfile('imdbFolder', dataset, [lower(dataset) '-seed-01'], 'imdb-seed-2_flipAug.mat');
    if ~exist(imdbFile, 'file')        
%         imdb = cub_get_database(dataDir);
        imdb = cubFlipAug_get_database(dataDir);
        
        imdbDir=fileparts(imdbFile);
        if ~isdir(imdbDir)
            mkdir(imdbDir);
        end
        save(imdbFile, '-struct', 'imdb') ;
    end
end
%% read pre-trained model and initialize network
netbasemodel = load('./initModel4demo.mat') ;
netbasemodel = netbasemodel.net;
netbasemodel.layers = netbasemodel.layers(1:34);
nDim = 512;
%% modify the pre-trained model to fit the current size/problem/dataset/architecture, excluding the final layer
batchSize = 16; % 32 for 224x224 image inpute, 8 for 448x448 input image size 
mopts.poolType='bilinear';

% only for pretrain multilayer perception
mopts.isPretrainFCs = false;
mopts.fc1Num = -1;
mopts.fc2Num = -1;
mopts.fcInputDim = 512*512;

% some usually fixed params
mopts.ftConvOnly = false;
mopts.use448 = true;
% mopts.use448 = false;
if mopts.use448
    inputImgSize = 448;
else
    inputImgSize = 224;
end
mopts.classifyType='SVM'; % or SVM or LR
mopts.initMethod='FroBiSVM'; % or 'pretrain' 'random' 'FroBiSVM' 'symmetricFullSVM'

% some parameters should be tuned
opts.train.batchSize = batchSize;
opts.train.learningRate = learningRate;
opts.train.weightDecay = weightDecay;
opts.train.momentum = 0.9 ;

% set the batchSize of initialization
mopts.batchSize = opts.train.batchSize;

% net=modifyNetwork_Shu(network, dataset, mopts);
if strcmp(netbasemodelName, 'VGG_M')
    endLayer=14;
    lastNchannel=512;
elseif strcmp(netbasemodelName, 'VGG_16')
    endLayer=30;
    lastNchannel=512;
elseif strcmp(netbasemodelName, 'VGG_19')
    endLayer=36;
    lastNchannel=512;
else % add resnets later
    error('unknown network');
end

% meta information for normalization
netbasemodel.meta.normalization.imageSize = [inputImgSize, inputImgSize, 3, netbasemodel.meta.normalization.imageSize(end)];
netbasemodel.meta.normalization.border = [29 29];        
fancyPCA = load('./fancyPCA.mat');
netbasemodel.meta.normalization.rgbVariance = fancyPCA.P* diag(0.1*fancyPCA.d(:));
%% add 1x1 conv layer initialized by PCA on conv5_3 feature
nclasses = num_classes;
m = 100; % reduced dimension
rDimBiClsNew = 8;
r = rDimBiClsNew;
orgDim = 512;

Pparam = load('CUB_bisvmEpoch102');
P = Pparam.U(:,1:m);
P = reshape(P,[1,1,size(P,1),size(P,2)]);
initParam = {{single(P), zeros(size(P,4),1,'single')}};
netbasemodel.layers{end+1} = struct('type', 'conv', ...
    'name', 'dimRedLayer', ...
    'weights', initParam, ...
    'stride', 1, ...
    'pad', 0, ...
    'learningRate', [0 0],...
    'weightDecay', [0 0], ...
    'precious', 1);

%% add the layer to the network
learnW = [0.01 0.02]; % [0.001 0.002]
lambda = 0.0001;
bisvmU = single(randn(m,num_classes*r)*0.01);
initFCparam = {{single(bisvmU), zeros(1,num_classes, 'single')}};
netbasemodel = addBisvm_posneg_UUregLayer(netbasemodel, mopts.classifyType,  initFCparam, rDimBiClsNew, num_classes, learnW, lambda);
%% setup to train network
%%{
% vl_simplenn_display(netbasemodel);
opts.imdbPath = fullfile('./imdbFolder', dataset, [lower(dataset) '-seed-01/imdb-seed-1.mat']);
% opts.imdbPath = './imdbFolder/CUB/cub-seed-01/imdb-seed-1.mat';

% opts.train.expDir = fullfile('./imdbFolder', dataset, 'exp', [dataset, '_', netbasemodelName, '_', mopts.classifyType, ...
%     '_', mopts.poolType, '_', int2str(inputImgSize), '_main020_initialFromPCA_randInitBisvm']);

opts.train.expDir = fullfile('./imdbFolder', dataset, 'exp', 'save4demo');

if ~isdir(opts.train.expDir)
    mkdir(opts.train.expDir);
end
opts.train.numSubBatches = 1 ;
opts.train.continue = true ;
opts.train.gpus = gpuId ;
%gpuDevice(opts.train.gpus); % don't want clear the memory
opts.train.prefetch = true ;
opts.train.sync = false ; % for speed
opts.train.cudnn = true ; % for speed
opts.train.numEpochs = numel(opts.train.learningRate) ;
%% modify the dataset for fast check
imdb = load(opts.imdbPath) ;
rng(777);
%% setup-II
% in case some dataset only has val/test
opts.train.val = union(find(imdb.images.set==2), find(imdb.images.set==3));
opts.train.train = [];

bopts = netbasemodel.meta.normalization;
bopts.numThreads = 12;
fn = getBatchWrapperModeAware(bopts) ;
opts.train.backPropDepth = inf; % could limit the backprop
prefixStr = [dataset, '_', netbasemodelName, '_', mopts.classifyType, '_', mopts.poolType, '_', int2str(inputImgSize), '_' ];

%% train
[netbasemodel, info] = cnntrainMain020_initialFromPCAonConv53(netbasemodel, imdb, prefixStr, fn, opts.train, 'conserveMemory', true);
% 136 -- increase posnegU penaly from 2 to 20


