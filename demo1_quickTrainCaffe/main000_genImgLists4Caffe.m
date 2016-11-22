clear
close all
clc;

%% generate image lists for training using caffe
dataDir = '../data/cub';
dataset = 'CUB';
imdbFile = fullfile('../imdbFolder', dataset, [lower(dataset) '-seed-01'], 'imdb-seed-1.mat');

imdb = load(imdbFile) ;
trainIdx = find(imdb.images.set==1);
testIdx = find(imdb.images.set==3);
rootPath = fullfile('/home/skong/data/BirdProject/matconvnetBilinear/data/cub/images/');
%% generate training image list
fprintf('generate training image list...\n');
filename = 'trainList.txt';
fn = fopen(filename, 'w');
for i = 1:length(trainIdx)
    fname = fullfile( rootPath, imdb.images.name{trainIdx(i)});
    label = imdb.images.label(trainIdx(i));
    fprintf(fn, '%s %d\n', fname, label);
    if mod(i,1000) == 0
        fprintf('\t%d\n',i);
    end
end
fclose(fn);
%% generate testing image list
fprintf('generate testing image list...\n');
filename = 'testList.txt';
fn = fopen(filename, 'w');
for i = 1:length(testIdx)
    fname = fullfile( rootPath, imdb.images.name{testIdx(i)});
    label = imdb.images.label(testIdx(i));
    fprintf(fn, '%s %d\n', fname, label);
    if mod(i,1000) == 0
        fprintf('\t%d\n',i);
    end
end
fclose(fn);



