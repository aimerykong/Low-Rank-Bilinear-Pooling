% quick setup
clear
close all
clc;

run(fullfile('../CompactBilinearPool/MatConvNet/vlfeat','toolbox','vl_setup'));
addpath(genpath('froBiCls'));
addpath(genpath('addLayerFunc'));
addpath(genpath('get_activations'));
addpath(genpath('get_batch'));
addpath(genpath('linear_classifier'));
addpath(genpath('prepare_dataset'));
addpath(genpath('layers'));
run(fullfile('matconvnetToolbox', 'matlab', 'vl_setupnn'));
addpath(genpath(fullfile('matconvnetToolbox','examples')));
addpath(genpath('../exportFig'));
addpath(genpath('layers'));


% addpath /home/skong2/caffeCUDNN/matlab
% caffe.reset_all();
% caffe.set_mode_gpu();
% caffe.set_device(2);
% model = '/home/skong2/data/BirdProject/bilinearDiagnosisVGG16/caffetrain2/archHinge_AveMaxAdjust_alpha0.0000.deploy';
% weights = '/home/skong2/data/BirdProject/bilinearDiagnosisVGG16/caffetrain2/snapshotAMA_lessTrained_alpha0.0000ExtmAug_WD0.1_iter_6000.caffemodel';
%
% if ~exist('netInit','var')
%     netInit = caffe.Net(model, weights, 'test');
% end
% W = netInit.params('fc', 1).get_data();
% b = netInit.params('fc', 2).get_data();
% W = W(:,2:end-1);
% b = b(2:end-1);
% save('caffeAMA_lessTrained_alpha0.0000ExtmAug_WD0.1_iter_6000.mat', 'W', 'b', 'weights', 'model')
%
% clear netInit
% caffe.reset_all();
% save('caffeGood_ExtmAugWithVal.mat', 'W', 'b', 'weights', 'model')
% load('caffeGood_ExtmAugWithVal.mat'); strfix = 'withVal';
% load('caffeGood_ExtmAug.mat'); strfix = 'noVal';
% load('caffe_ExtmAug.mat'); strfix = 'noVal';
%% configuration
% dataset: 'CUB' (bird), 'MIT' (indoor scene), 'FMD' (fclikr material),
% 'DTD' (describable texture), 'aircraft' (ox), 'cars' (stanford)
dataset = 'CUB';

% network: VGG_M, VGG_16, VGG_19, resnet-152, resnet-50, resnet-101
netbasemodelName = 'VGG_16';

gpuId=3;
gpuDevice(gpuId);

% learningRate=[ones(1, 2000)*1E-4];
learningRate=[ones(1, 1000)*0.01];
% weightDecay: usually use the default value
weightDecay=0.0005;
%% prepare data
% dataset: 'CUB', 'MIT', 'DTD', 'aircrafts', 'cars'
if strcmp(dataset, 'CUB')
    num_classes = 200;
    dataDir = '../CompactBilinearPool/MatConvNet/data/cub';
    imdbFile = fullfile('imdbFolder', dataset, [lower(dataset) '-seed-01'], 'imdb-seed-1.mat');
    if ~exist(imdbFile, 'file')
        imdb = cub_get_database(dataDir);
        
        imdbDir=fileparts(imdbFile);
        mkdir(imdbDir);
        save(imdbFile, '-struct', 'imdb') ;
    end
end
%% read pre-trained model and initialize network
% well-trained model with positive constraint
% netbasemodel = load('./imdbFolder/CUB/exp/CUB_VGG_16_SVM_bilinear_448_main012_20rank_init020learnablePower_bicls_BN3/version10/CUB_VGG_16_SVM_bilinear_448_net-epoch-17.mat') ;
% netbasemodel = netbasemodel.net;
% netbasemodel.layers = netbasemodel.layers(1:34);

% less trained model using linvear SVM
% netbasemodel = load('./imdbFolder/CUB/exp/CUB_VGG_16_SVM_bilinear_448_final6_020RootOnlyFullBCNN_withBN3_goodway/CUB_VGG_16_SVM_bilinear_448_net-epoch-102.mat') ;
% netbasemodel = netbasemodel.net;
% netbasemodel.layers = netbasemodel.layers(1:33);

% well-trained model using positive&negative bisvm
netbasemodel = load('./imdbFolder/CUB/exp/CUB_VGG_16_SVM_bilinear_448_main017_20rank_learnablePower_bisvm/version6_simultanesouTrain_smallLR_longer_leakyRampLoss/CUB_VGG_16_SVM_bilinear_448_net-epoch-171.mat') ;

netbasemodel = netbasemodel.net;
bisvmLayer = netbasemodel.layers{36};
netbasemodel.layers = netbasemodel.layers(1:35); % 
netbasemodel = addBilinear(netbasemodel); % bilinear pooling layer
netbasemodel = vl_simplenn_tidy(netbasemodel) ;
vl_simplenn_display(netbasemodel);
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

vl_simplenn_display(netbasemodel);
netInfo = vl_simplenn_display(netbasemodel);
outputSize = 512*512;
%% get activations from bilinear pooling layer for fast test
[valFV, valY] = get_bilinearFeature_main018testOnly(...
    dataset, netbasemodelName, mopts.poolType, mopts.classifyType, mopts.initMethod, mopts.use448, netbasemodel, batchSize);
%% spectrum of learned bilinear SVM's
fprintf('spectrum of learned pos-neg bsvm\n');
W = bisvmLayer.weights{1};
W = reshape(W, [size(W,1),  rDimBiCls, num_classes]); % 512 x rDim x nclasses
b = bisvmLayer.weights{2};

orgDim = size(W,1);
dimFixed = rDimBiCls/2;
nclasses = num_classes;
UList = zeros(orgDim, orgDim,nclasses);
VList = zeros(orgDim, orgDim,nclasses);
singularList = zeros(orgDim, nclasses);

posUList = zeros(orgDim, dimFixed, nclasses);
posSingularList = zeros(dimFixed, nclasses);
negUList = zeros(orgDim, dimFixed, nclasses);
negSingularList = zeros(dimFixed, nclasses);
W_lsvm = zeros(orgDim, orgDim, nclasses);
Wpos = zeros(orgDim,dimFixed,nclasses);
Wneg = zeros(orgDim,dimFixed,nclasses);

for c = 1:nclasses
    curWpos = W(:,1:rDimBiCls/2,c)*W(:,1:rDimBiCls/2,c)';
    curWneg = W(:,1+rDimBiCls/2:end,c)*W(:,1+rDimBiCls/2:end,c)';
    curW = curWpos-curWneg;
    [curU, curD, curV] = svd( curW );        
    curd = diag(curD);    
    %% positive&negative
    sgnU = sign(curU(1,:));
    sgnV = sign(curV(1,:));
    a = sgnV.*sgnU;
    curd(a==-1) = curd(a==-1)*(-1);
    curd = curd(:);
    for j = 1:length(a)
        if a(j)==-1
            curV(:,j) = -1*curV(:,j);
        end
    end
    
    [dsort, didx] = sort(curd,'descend');
    UList(:,:,c) = curU(:,didx);
    VList(:,:,c) = curV(:,didx);
    singularList(:,c) = curd(didx);
    
    curU = UList(:,:,c);
    curd = singularList(:,c);
    posUList(:,:,c) = curU(:, 1:dimFixed);
    posSingularList(:,c) = curd(1:dimFixed);
    negUList(:,:,c) = curU(:, end-dimFixed+1:end);
    negSingularList(:,c) = curd( end-dimFixed+1:end);
    
    Wpos(:,:,c) = posUList(:,:,c)*diag(posSingularList(:,c).^0.5);
    Wneg(:,:,c) = negUList(:,:,c)*diag((-1*negSingularList(:,c)).^0.5);
    %W_lsvm(:,:,c) = curW; % good
    W_lsvm(:,:,c) = Wpos(:,:,c)*Wpos(:,:,c)'-Wneg(:,:,c)*Wneg(:,:,c)'; % good
    %%
    if ~isempty(find(posSingularList(:,c)<-10e-7)) || ~isempty(find(negSingularList(:,c)>10e-7))             
        fprintf('attention class-%d -- ~isempty(find(posSingularList(:,i)<0)) || ~isempty(find(negSingularList(:,i)>0))\n', c);
    end
%     [curU, curD] = svd( W_lsvm(:,:,c) );
%     SingularVecList(:,:,c) = curU;
%     singularValList(:,c) = diag(curD);
end

singularListTMP = [singularList(1:dimFixed,:); singularList(end+1-dimFixed:end,:)];
meanList = mean(singularListTMP,2);
stdList = std(singularListTMP');

figure(1);
errorbar( 1:length(meanList), meanList(:)', stdList(:)', '-o', 'MarkerSize',2, 'MarkerEdgeColor', 'red', 'MarkerFaceColor', 'red' );
title('sorted eigen values with std for all classes');
xlabel('index of sorted singular value');
ylabel('eigen value'); 
grid on;

fprintf('done\n');
%% fast verify the model
W_lsvm_tmp = reshape(W_lsvm, [orgDim*orgDim, nclasses]);
score = bsxfun(@plus, W_lsvm_tmp'*valFV, b(:));
[~,predLabel] = max(score,[],1);
grndLabel = valY(:);
acc = mean(grndLabel(:)==predLabel(:));
fprintf('fast verification of the model: acc=%.6f\n', acc);
%% matrices co-factorization without constraints for positive and negative parts sharing dimRed
% Wi ~ P*Ui,   for i=1,...,nclasses
% Wi -- d x d   here d=512
% P  -- d x m   here m is in dimRedList 
% Ui -- m x r   here r is in rankList

% save('coDecomp_WposWneg.mat', 'Wpos', 'Wneg', 'b');

flag_optimization = false;
dimRedList = 10:10:300; % m in P of size dxm
rankList = 2:2:20; % the total rank including positive and negative parts

accMat = zeros(length(rankList),length(dimRedList));
paramSizeMat = zeros(length(rankList),length(dimRedList));
curW = reshape(W_lsvm, [orgDim,orgDim,nclasses]);
curW = reshape(curW, [orgDim*orgDim, nclasses]);

for r = rankList
    WposTMP = Wpos(:,1:r/2,:);
    WnegTMP = Wneg(:,end-r/2+1:end,:);
    for m = dimRedList
%         if r > m
%             break;
%         end
        Wcat = cat(3, WposTMP,WnegTMP);
        [Ucat, P, Wcat_approx] = matricesCoFactorization(Wcat, m, flag_optimization);
        Wpos_approx = Wcat_approx(:,:,1:nclasses);
        Wneg_approx = Wcat_approx(:,:,1+nclasses:end);
        curW = zeros(orgDim,orgDim,nclasses);
        for i = 1:nclasses
            curW(:,:,i) = Wpos_approx(:,:,i)*Wpos_approx(:,:,i)' - Wneg_approx(:,:,i)*Wneg_approx(:,:,i)'; 
        end
        %% classification
        curW = reshape(curW, [orgDim*orgDim, nclasses]);
        score = bsxfun(@plus, curW'*valFV, b(:));
        [~,predLabel] = max(score,[],1);
        grndLabel = valY(:);
        acc = mean(grndLabel(:)==predLabel(:));
%         rDim=512;m=512;r=20;nclasses=200;
        paramSize = orgDim*m + nclasses*m*r;
        paramSize = paramSize * 4/(1024^2);
        
        fprintf('rank-%03d coDimRed-%03d on testset: acc=%.6f, paramSize: %.3f MB\n', r, m, acc, paramSize);
        accMat(r/2,m/2) = acc; % accMat = zeros(length(rankList),length(dimRedList));
        paramSizeMat(r/2,m/2) = paramSize;        
    end
end
%% store and visualize results 
save('acc_dimRed_rank_sharedDimParam4PosNeg_bestTrainedModel.mat', ...
    'rankList', 'dimRedList', 'Wpos', 'Wneg',...
    'flag_optimization', 'paramSizeMat', 'accMat', 'grndLabel', 'b');


showMax_dim = 150;
withinMaxDimIdxs = find(dimRedList<=showMax_dim);
% withinMaxDimIdxs = withinMaxDimIdxs(1:end);
figure(2);
% subplot(1,2,1)
accMatShow = accMat;
accMatShow = accMatShow(rankList/2,:);
accMatShow = accMatShow(:,dimRedList(withinMaxDimIdxs)/2);
bar3(accMatShow);
xlabel('dimRed - m');
ylabel('rank - r');
zlabel('acc');
ax = gca;
ax.YTick = 1:length(rankList);
ax.YTickLabel = rankList;
ax.XTick = 1:length(dimRedList(withinMaxDimIdxs));
ax.XTickLabel = dimRedList(withinMaxDimIdxs);
title('acc vs. rank & dimRed');
az = -60; % Azimuth
el = 20; % Elevation
view(az, el);

figure(3);
% subplot(1,2,2);
paramSizeMatShow = paramSizeMat;
paramSizeMatShow = paramSizeMatShow(rankList/2,:);
paramSizeMatShow = paramSizeMatShow(:,dimRedList(withinMaxDimIdxs)/2);
bar3(paramSizeMatShow);
xlabel('dimRed - m');
ylabel('rank - r');
zlabel('size in MB');
ax = gca;
ax.YTick = 1:length(rankList);
ax.YTickLabel = rankList;
ax.XTick = 1:length(dimRedList(withinMaxDimIdxs));
ax.XTickLabel = dimRedList(withinMaxDimIdxs);
title('paramSize vs. rank & dimRed');
view(az, el);

[a,b] = max(accMatShow(:));
[y,x] = ind2sub(size(accMatShow),b);
fprintf('\nmax acc:%.4f, paramSize: %.3fMB, rank:%d in total, dimRed:%d\n', accMatShow(y,x), paramSizeMatShow(y,x), rankList(y), dimRedList(x));
disp(a)
disp(accMatShow(y,x)); % rankList(y), dimRedList(x)
disp( accMat(rankList(y)/2, dimRedList(x)/2 ) ); % rankList(y), dimRedList(x)

%% leave blank