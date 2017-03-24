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
% vl_simplenn_display(netbasemodel);
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
vl_simplenn_display(netbasemodel);
desiredTestList = [1:1:100];
%% read an image
for imgIdx = desiredTestList % 220:4:length(tsIdx) % 220
    tsImgID = tsIdx(imgIdx); % end
    grndLabel = imdb.images.label(tsImgID) ;
%     imageName = strcat('../data/cub/images/', imdb.images.name(tsImgID)) ;
    imageName = strcat('./cub/images/', imdb.images.name(tsImgID)) ;
    [imo, imBackup] = fetchCUB4BackpropAct(imageName, mode, optsImg) ;
    fprintf('simplificationTestCls%d_Img%d.fig\n', grndLabel, imgIdx);
    
    [validFlag, curMASK] = hirarchicalSimplifyImage( netbasemodel, gpuArray(imo), grndLabel);
    %% save    
    if validFlag
        export_fig(sprintf('simplificationTestCls%d_Img%d.fig', grndLabel, imgIdx));
    end
end
%% leave blank
