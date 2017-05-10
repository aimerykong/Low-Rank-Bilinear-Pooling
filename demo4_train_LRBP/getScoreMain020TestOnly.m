function [Gram] = getScoreMain020TestOnly...
    (imdb, batchIds, netGpu, getBatchFunc, batchSize)


if nargin<5
    batchSize=128;
end

% if net is larger than 100k, the downsample it to 10k
% mainly to be friendly to MIT Places dataset
if numel(batchIds) > 100000
    batchIds=batchIds(randsample(numel(batchIds), 10000));
    fprintf(['Warning: In getNetLastActivations, input batch too large',...
        ', downsampled to 10K']);
end

Gram = 0;

imList = single(zeros(448,448,3,ceil(numel(batchIds)/batchSize)));
for i=1:ceil(numel(batchIds)/batchSize)
    if mod(i,100)==0
        fprintf('In getNetLastActivations, reading and get activations of batch %d\n',i);
    end
    
    inter=getInter(i, batchSize, numel(batchIds));
    [curImages, labels]=getBatchFunc(imdb, batchIds(inter),'test');
    imList(:,:,:,i) = curImages;
    % prefetch
    getBatchFunc(imdb, batchIds(getInter(i+1, batchSize, numel(batchIds))),'test');
    
    res = vl_simplenn(netGpu, gpuArray(curImages),[],[], 'conserveMemory', true, 'mode', 'test'); %acc=0.818778
    
    final_resp=gather(res(end).x);
    %% biCls
    hw = size(final_resp,1);
    final_resp = reshape(final_resp, [size(final_resp,1)*size(final_resp,2), size(final_resp,3)]);
    final_resp = final_resp'*final_resp;
    Gram = Gram + (1/hw)*final_resp;
end





