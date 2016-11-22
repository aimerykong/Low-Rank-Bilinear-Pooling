function [Gram] = get_score_for_PCAinitP(dataset, network, endLayer, classificationType, initMethod, use448, net, batchSize)

useFewShot = false; % this is only for few shots learning

% two calling methods:
% 1. get_activations_dataset_network_layer('MIT', 'VGG_M', 13);
% 2. get_activations_dataset_network_layer(dataset, network, 'pcaConv', use448, net, batchSize);

%% filling in absent values
if nargin<3,
    error('too few input arguments');
end
if nargin<4, use448=false;
    fprintf('default value use448=false\n');
end

if nargin<5
    net=load(consts(dataset, network));
    % implicitly assume that endLayer is a number
    net.layers=net.layers(1:endLayer);
    fprintf('default net read from (dataset, network, endLayer)\n');
end
if nargin<6,
    batchSize=8;
    fprintf('default batchSize=8\n');
end

% use448 could be an int32, indicates specific resolution
if isa(use448, 'logical'),
    use448=int32(use448*224+224);
end

% from endLayer to deduce confId. Note that endLayer could be 'char'
if isa(endLayer, 'numeric'),
    endLayer=num2str(endLayer);
end
confId=endLayer;

%% feature output file name
vl_simplenn_display(net);

imdb = load(sprintf('../imdbFolder/%s/%s-seed-01/imdb-seed-1.mat',dataset, lower(dataset)));

imdb.imageDir = ['.' imdb.imageDir];

train = imdb2set(imdb.images.set, 1);

% val=imdb2set(imdb.images.set, 0);
% % support for few shot learning
% if useFewShot
%     train = few_train(imdb, 1);
% end

% debug resize the image to be 448*448
net.normalization.imageSize=double([use448 use448 3]);
net.normalization.averageImage = net.meta.normalization.averageImage;
net.normalization.rgbVariance = net.meta.normalization.rgbVariance;

bopts = net.normalization ;
bopts.numThreads = 12;
getBatch = getBatchWrapperModeAware_car(bopts);

% get the features
netGpu = vl_simplenn_move(net, 'gpu');
%         [trainFV, trainY]=getNetLastActivationsMain014(imdb, train, netGpu, getBatch, batchSize);

[Gram] = getScoreMain001GramTrain(imdb, train, netGpu, getBatch, batchSize);


