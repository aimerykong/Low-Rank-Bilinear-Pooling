%% quick setup
clear
close all
clc;

addpath('../');
addpath(genpath('../froBiCls'));
addpath(genpath('../addLayerFunc'));
addpath(genpath('../get_activations'));
addpath(genpath('../get_batch'));
addpath(genpath('../linear_classifier'));
addpath(genpath('../prepare_dataset'));
addpath(genpath('../layers'));
addpath(genpath(fullfile('../matconvnetToolbox','examples')));

run(fullfile('../../CompactBilinearPool/MatConvNet/vlfeat','toolbox','vl_setup'));
run(fullfile('../matconvnetToolbox', 'matlab', 'vl_setupnn'));

addpath(genpath('../../exportFig'));
%% configuration
% dataset: 'CUB' (bird), 'MIT' (indoor scene), 'FMD' (fclikr material), 
% 'DTD' (describable texture), 'aircraft' (ox), 'cars' (stanford)
dataset = 'CUB';

% network: VGG_M, VGG_16, VGG_19, resnet-152, resnet-50, resnet-101
netbasemodelName = 'VGG_16';

gpuId = 3;
gpuDevice(gpuId);

learningRate = [ones(1, 80)*0.01, ones(1, 40)*0.001, ones(1, 20)*0.0001];
% weightDecay: usually use the default value
weightDecay=0.0005;
%% prepare data
% dataset: 'CUB', 'MIT', 'DTD', 'aircrafts', 'cars'
if strcmp(dataset, 'CUB')
    num_classes = 200;
    dataDir = './data/cub';
    imdbFile = fullfile('../imdbFolder', dataset, [lower(dataset) '-seed-01'], 'imdb-seed-1.mat');
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
elseif strcmp(dataset, 'aircraft')
    dataDir = '../data/aircraft';
    imdbFile = fullfile('../imdbFolder', dataset, [lower(dataset) '-seed-01'], 'imdb-seed-1.mat');    
    num_classes = 100;
    if ~exist(imdbFile, 'file')        
        imdb = aircraft_get_database(dataDir, 'variant'); % family (70), manufacturer (30), variant (100)
        
        imdbDir = fileparts(imdbFile);
        if ~isdir(imdbDir)
            mkdir(imdbDir);
        end
        save(imdbFile, '-struct', 'imdb') ;
    end
elseif strcmp(dataset, 'cars')
    dataDir = '../data/cars';
    imdbFile = fullfile('../imdbFolder', dataset, [lower(dataset) '-seed-01'], 'imdb-seed-1.mat');    
    num_classes = 196;
    if ~exist(imdbFile, 'file')        
        imdb = cars_get_database(dataDir, 'variant'); % family (70), manufacturer (30), variant (100)
        
        imdbDir = fileparts(imdbFile);
        if ~isdir(imdbDir)
            mkdir(imdbDir);
        end
        save(imdbFile, '-struct', 'imdb') ;
    end
end
%% read pre-trained model and initialize network
netbasemodel = load('/home/skong2/data/BirdProject/matconvnetBilinear/imdbFolder/CUB/exp/CUB_VGG_16_SVM_bilinear_448_final5FullBCNN/CUB_VGG_16_SVM_bilinear_448_net-epoch-13.mat');
netbasemodel = netbasemodel.net;
netbasemodel = vl_simplenn_tidy(netbasemodel) ;
vl_simplenn_display(netbasemodel);
endLayer = 30;
% netbasemodel.layers = netbasemodel.layers(1:30);
% vl_simplenn_display(netbasemodel);

%% modify the pre-trained model to fit the current size/problem/dataset/architecture, excluding the final layer
batchSize = 1; % 32 for 224x224 image inpute, 8 for 448x448 input image size 
rDimBiCls = 20; % reduced dimension by bilinear SVM
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
mopts.initMethod='pretrain'; % or 'pretrain' 'random' 'FroBiSVM' 'symmetricFullSVM'

% some parameters should be tuned
opts.train.batchSize = batchSize;
opts.train.learningRate = learningRate;
opts.train.weightDecay = weightDecay;
opts.train.momentum = 0.9 ;

% set the batchSize of initialization
mopts.batchSize = opts.train.batchSize;

% fancy PCA
fancyPCA = load('./fancyPCA.mat');
netbasemodel.meta.normalization.rgbVariance = fancyPCA.P* diag(0.1*fancyPCA.d(:));
netbasemodel.meta.normalization.imageSize = [inputImgSize, inputImgSize, 3, netbasemodel.meta.normalization.imageSize(end)];

netbasemodel.layers = netbasemodel.layers(1:endLayer);
netbasemodel = vl_simplenn_tidy(netbasemodel) ;
vl_simplenn_display(netbasemodel);
%% fetch data
[Gram] = get_score_for_PCAinitP( dataset, netbasemodelName, mopts.poolType, mopts.classifyType, mopts.initMethod, mopts.use448, netbasemodel, batchSize);

[U,S,V] = svd(Gram);
s = diag(S);
figure;
plot(1:length(s), s, 'r-.');
rDim = 100; disp( sum(s(1:rDim))/sum(s) )


save('bird_VGG16_conv53', 'Gram', 'U', 's', 'S');


