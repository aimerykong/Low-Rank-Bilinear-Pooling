% quick setup
clear
close all
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

learningRate=[ones(1, 1000)*0.01];
% weightDecay: usually use the default value
weightDecay=0.0005;
%% prepare data
% dataset: 'CUB', 'MIT', 'DTD', 'aircrafts', 'cars'
if strcmp(dataset, 'CUB')
    num_classes = 200;
    imdbFile = fullfile('imdbFolder', dataset, [lower(dataset) '-seed-01'], 'imdb-seed-1.mat');
end
%% read pre-trained model and initialize network
netbasemodel = load('./initModel4demo.mat') ;
netbasemodel = netbasemodel.net;
netbasemodel.layers = netbasemodel.layers(1:34);
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
mopts.initMethod='FroBiSVM'; % or 'random' 'FroBiSVM' 'symmetricFullSVM'

% some parameters should be tuned
opts.train.batchSize = batchSize;
opts.train.learningRate = learningRate;
opts.train.weightDecay = weightDecay;
opts.train.momentum = 0.9 ;

% set the batchSize of initialization
mopts.batchSize = opts.train.batchSize;

netInfo = vl_simplenn_display(netbasemodel);
outputSize = 512*512;
%% fetch data
[Gram] = get_score_main020testOnly(...
    dataset, netbasemodelName, mopts.poolType, mopts.classifyType, mopts.initMethod, mopts.use448, netbasemodel, batchSize);

[U,S,V] = svd(Gram);
s = diag(S);
figure;
% plot(1:length(s), s, 'r-.');

rDim = 100; disp( sum(s(1:rDim))/sum(s) )
save('CUB_bisvmEpoch102', 'Gram', 'U', 's', 'S');



