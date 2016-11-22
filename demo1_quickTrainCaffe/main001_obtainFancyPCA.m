% fancy PCA for data augmentation
% read all images
% get mean (or default) and covariance matrix
% compute PCA

% clear
close all
clc;
%% read all images and approximately compute mean and covariance
dataDir = '../data/cars';
dataset = 'cars';
imdbFile = fullfile('../imdbFolder', dataset, [lower(dataset) '-seed-01'], 'imdb-seed-1.mat');

imdb = load(imdbFile) ;

avgRGB_ImageNet = [123.6800, 116.7790, 103.9390]';
avgRGB_AircraftApprox = zeros(3,1);
avgRGB_AircraftReal = zeros(3,1);
DBpath = imdb.imageDir;
pixelCount = 0;
imCount = 0;
Gram = 0;
for i = 1:length(imdb.images.name)
    if mod(i,100) == 0
        fprintf('\timage-%d...\n',i);
    end
    imCount = imCount + 1;
    
    im = imread( fullfile(DBpath, imdb.images.name{i}) );
    sz = size(im);
    HW = sz(1)*sz(2);
    if length(sz) == 2
        im = repmat(im, [1,1,3]);
    end
    %% covariance matrix of the current image
    im = reshape(im, [HW, 3]);
    im = im'; % [3, HW]
    im = double(im);
    avgRGB_AircraftApprox = avgRGB_AircraftApprox + mean(im,2);
    avgRGB_AircraftReal = avgRGB_AircraftReal + sum(im,2);
    im = bsxfun(@minus, im, avgRGB_ImageNet);
    im = im*im'/HW;
    Gram = Gram + im;
    
    pixelCount = pixelCount + HW;    
end
avgRGB_AircraftApprox = avgRGB_AircraftApprox / imCount;
Gram = Gram / pixelCount;
avgRGB_AircraftReal = avgRGB_AircraftReal/pixelCount;

%% PCA
[P, d] = eig(Gram);
d = diag(d);

save('fancyPCA.mat', 'pixelCount', 'Gram', ...
    'avgRGB_AircraftReal', 'avgRGB_ImageNet', 'avgRGB_AircraftApprox', ...
    'imCount', 'pixelCount', 'P', 'd');


