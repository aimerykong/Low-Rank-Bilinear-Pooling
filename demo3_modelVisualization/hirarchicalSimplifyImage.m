function [validFlag, curMASK] = hirarchicalSimplifyImage( net, im, grndLabel, superPixelRegionSizeList )
%
%
% requirement --
%   1. The input image and the network model can be moved to GPU in advance to
%       accelerate the whole process
%   2. vlfeat should be setup before running the function
% Note that --
%   1. the output layer (last layer) can be modified to produce confidence
%       score; or if maximum margin classifier is used, the output should
%       transformed into confidence score (see line 32~35).
%
% Shu Kong @ UCI
% March 2017

%% default configuration
% run(fullfile('../../CompactBilinearPool/MatConvNet/vlfeat','toolbox','vl_setup'));
if ~exist('superPixelRegionSizeList', 'var')
    superPixelRegionSizeList = [200, 150, 100, 50]; %% trade-off between result quality and speed -- make it finer for better quality
end
REGULARIZER = 100; % the larger the more rigid rectangular
curMASK = ones(size(im,1), size(im,2));
validFlag = 1;
%% first pass to check whether the image is valid for simplification
res = my_simplenn(net, im, [], [], ...
    'accumulate', 0, ...
    'mode', 'test', ...
    'conserveMemory', 1, ...
    'cudnn', true) ;

outputScore = gather(res(end).x); %% maximum margin output?
outputScore = outputScore - min(outputScore);
outputScore = exp(outputScore);
outputScore = outputScore ./ sum(outputScore); % transformed into confidence score
[~, predLabel] = max(outputScore);
if grndLabel~=predLabel
    validFlag = 0;
    return;
end

orgImg = bsxfun(@plus, im, net.meta.normalization.averageImage);
figure(1);
numColumn = length(superPixelRegionSizeList) + 1;
subplot(1, numColumn, 1);
imshow(uint8(orgImg)); title('orgImg');

%% iteratively simplify the input image with oversegment at different scales
maskList = cell(1, length(superPixelRegionSizeList)+1);
maskList{1} = ones(size(im,1), size(im,2));
scaleIdx = 1;
for REGIONSIZE = superPixelRegionSizeList % size of superpixel
    fprintf('at scale-%d...', REGIONSIZE);
    scaleIdx = scaleIdx + 1;    
    %% initialization using previous output
    maskList{scaleIdx} = maskList{scaleIdx-1}; % initialize the mask using the previous output
    curMASK = maskList{scaleIdx};
    im2seg = im .* repmat(maskList{scaleIdx}, [1,1,3]);
    SEGMENTS = vl_slic(gather(im2seg), REGIONSIZE, REGULARIZER); % over-segment the image using slic
    SEGMENTS = SEGMENTS .* uint32(maskList{scaleIdx}); %  masking the segments using the current valid image region
    
    uniqueLabel = unique(SEGMENTS); 
    containZero = find(uniqueLabel==0);
    Nsegment = length(uniqueLabel); % number of segments excluding zero labeled segments
    segKeptFlagList = ones(1, Nsegment); % recording whether the segment is kept during iteration
    if ~isempty(containZero)
        segKeptFlagList(containZero) = 0;
    end
    curSEGMENTS = SEGMENTS;     % 
    
    MAXITER = 300;
    iter = 0;
    predLabel = grndLabel;
    while grndLabel == predLabel && iter < MAXITER
        iter = iter + 1;
        fprintf('\n\tround-%d (#seg:%d)...', iter, sum(segKeptFlagList));
        idxList = find(segKeptFlagList==1);
        scoreMat = zeros( length(outputScore), length(idxList));
        for i = 1:length(idxList)            
            mask = curMASK;
            mask( curSEGMENTS == uniqueLabel(idxList(i)) ) = 0;
            imoTMP = im.* repmat(mask,[1,1,3]);
            
            res = my_simplenn(net, gpuArray(imoTMP), [], [], ...
                'accumulate', 0, ...
                'mode', 'test', ...
                'conserveMemory', 1, ...
                'cudnn', true) ;
            outputScoreTMP = gather(res(end).x); %% maximum margin output
            outputScoreTMP = outputScoreTMP - min(outputScoreTMP);
            outputScoreTMP = exp(outputScoreTMP);
            outputScoreTMP = outputScoreTMP ./ sum(outputScoreTMP); % transformed into softmax score
            scoreMat(:,i) = outputScoreTMP(:);
        end        
        
        [~, maxwhere] = max(scoreMat(grndLabel,:));
        outputScoreTMP = scoreMat(:,maxwhere);
        [~, predLabel] = max(outputScoreTMP);
        if predLabel == grndLabel
            curMASK(curSEGMENTS == uniqueLabel(idxList(maxwhere))) = 0;
            segKeptFlagList(idxList(maxwhere)) = 0;
        else
            fprintf('done!\n');
        end
    end
    maskList{scaleIdx} = curMASK;
    %% show
    tmpImg = orgImg .* repmat(curMASK,[1,1,3]);
    subplot(1, numColumn, scaleIdx);
    imshow(uint8(tmpImg)); title(sprintf('scale-%d', superPixelRegionSizeList(scaleIdx-1)));
end