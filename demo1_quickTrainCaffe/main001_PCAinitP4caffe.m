% quick setup
% clear
close all
clc;

addpath ../caffe-20160312/matlab

% caffe.reset_all();
caffe.set_mode_gpu();
% caffe.set_device(1);

model = './archBird.deploy';
weights = './initModel_rDim100.caffemodel';

if ~exist('netInit','var')
    netInit = caffe.Net(model, weights, 'test');
end

%% read images
dataset = 'CUB';
imdbPath = fullfile('../imdbFolder', dataset, [lower(dataset) '-seed-01/imdb-seed-1.mat']);
imdb = load(imdbPath) ;
imdb.imageDir = '../data/cub/images';

trList = find(imdb.images.set==1);
imSize = 448;

Gram = 0;
meanData = caffe.io.read_mean('./meanImg.binaryproto');
for i = 1:length(trList)
    if mod(i,1000) == 0
        fprintf('\t%d/%d...\n', i, length(trList));
    end
    imID = trList(i);
    label = imdb.images.label(imID);
    imName = fullfile( imdb.imageDir, imdb.images.name{imID} );
    
    width = size(meanData,1);
    height = size(meanData,2);
    imData = caffe.io.load_image(imName);
    if size(imData,3) == 1
        imData = repmat(imData,[1,1,3]);
    end
    imData = imresize(imData, [width, height]);
    imData = imData - meanData;
    imData = imData( (size(imData,1)-imSize)/2:(size(imData,1)-imSize)/2+imSize-1, (size(imData,2)-imSize)/2:(size(imData,2)-imSize)/2+imSize-1,:);
    res = netInit.forward({imData});
    
    conv53 = netInit.blobs('conv5_3').get_data();    
    conv53 = reshape(conv53, [size(conv53,1)*size(conv53,2), size(conv53,3)]);
    
    hw = size(conv53,1);
    conv53 = conv53'*conv53;
    Gram = Gram + (1/(hw^2))*conv53;
end

%% PCA initialzation
[U,S,V] = svd(Gram);
s = diag(S);
figure;
plot(1:length(s), s, 'r-.');
rDim = 100; disp( sum(s(1:rDim))/sum(s) )

W = netInit.params('conv_dimRed',1).get_data();
newW = U(:,1:size(W,4))*diag(s(1:rDim).^(-0.5));
% newW = U(:,1:size(W,4));
newW = reshape(newW, size(W));
netInit.params('conv_dimRed', 1).set_data(newW);

save('bird_VGG16_conv53_4caffe', 'Gram', 'U', 's', 'S');
netInit.save('PCA_initCaffemodel_scaled.caffemodel');

