function [activations, label_out] = getScoreMain016TestOnly...
                         (imdb, batchIds, netGpu, getBatchFunc, batchSize)
    isFirst=1;
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
    
    label_out=zeros(numel(batchIds),1);
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
        
        label_out(inter)=labels;
        
%         curImages = bsxfun(@minus, curImages, netGpu.normalization.averageImage); % mean subtraction!!!
%         netGpu.layers{end-1}.precious = 1;
%         netGpu.layers{end}.class = labels ; 
        res = vl_simplenn(netGpu, gpuArray(curImages),[],[], 'conserveMemory', true, 'mode', 'test'); %acc=0.818778
%         res = my_simplenn(netGpu, gpuArray(curImages),[],[], 'conserveMemory', true); %acc=0.818778
%         res = my_simplenn(netGpu, gpuArray(curImages),[],[], 'conserveMemory', true, 'mode', 'test'); %acc=0.766655
%         res = my_simplenn(netGpu, gpuArray(curImages),[],[], 'conserveMemory', true, 'accumulate', 0); %acc=0.822575
%         res = my_simplenn(netGpu, gpuArray(curImages),[],[], 'conserveMemory', true, 'accumulate', 0, 'sync', 0, 'cudnn', 1); % acc=0.822575
%         res = my_simplenn(netGpu, gpuArray(curImages),[],[], 'conserveMemory', true, 'backPropDepth', Inf, 'accumulate', 0, 'sync', 0, 'cudnn', 1); % acc=0.822230
%         res = vl_simplenn(netGpu, gpuArray(curImages),[],[], 'mode', 'test', ...
%             'conserveMemory', true, 'backPropDepth', Inf, 'accumulate', 0, 'sync', 0, 'cudnn', 1); % acc=0.822230
        
%         res = vl_simplenn(netGpu, gpuArray(curImages), [], [], ...
%             'accumulate', 0, ...
%             'mode', 'test', ...
%             'conserveMemory', 1, ...
%             'backPropDepth', Inf, ...
%             'sync', 0, ...
%             'cudnn', 1) ; % acc=0.770452
        
        final_resp=gather(res(end).x);
        
        %% max pooling
        %{
        final_resp = reshape(final_resp, [size(final_resp,1)*size(final_resp,2), size(final_resp,3)]);
        final_resp = final_resp';
        TMP = zeros(size(final_resp,1), size(final_resp,1), size(final_resp,2));
        for iii = 1:size(final_resp,2)
            TMPP = final_resp(:,iii);
            TMP(:,:,i) = TMPP*TMPP';
        end
        TMP = max(TMP,[],3);
        final_resp = single(TMP);
        %}
        %% ave pooling
        %{
        final_resp = reshape(final_resp, [size(final_resp,1)*size(final_resp,2), size(final_resp,3)]);
        final_resp = final_resp';
        final_resp = final_resp*final_resp';
        %}       
        %% spatial-wise normalization
        %{
        a = 2;
        b = 4;
        S = sum(final_resp, 3 ); % HxWxC
        z = sum(S(:).^a);
        z = z^(1/a);
        S = (S./z).^(1/b);
        final_resp = final_resp .* repmat(S, [1,1,size(final_resp,3)]);
        %}
        %% channel-wise normalization
        %{
        K = 512;
        eps = 0.00001;
        S = final_resp; % HxWxC
        H = size(S,1);
        W = size(S,2);
        S(S>0) = 1;
        S = sum(S,2);
        S = sum(S,1);
        S = S/(H*W);
        q = sum(S(:));
        I = log( (K*eps+q) ./ (eps+S) );
        %I = reshape(I, [1,1,length(I)]);
        final_resp = final_resp .* repmat(I, [H, W, 1]);
        %}
        %% biCls
        final_resp = reshape(final_resp, [size(final_resp,1)*size(final_resp,2), size(final_resp,3)]);
        final_resp = 1*final_resp';
% %         final_resp = 1000*final_resp';
%         final_resp = final_resp*final_resp';
        
        %%        
        % assign the response to output structure
        if isFirst
            isFirst=0;
            dims=getDimOfBlob(final_resp);
            activations_out=zeros(dims{:}, numel(batchIds), 'single');
%             activations_out=zeros(dims{:}^2, numel(batchIds), 'single');
%             activations_out=zeros(512^2, numel(batchIds), 'single');
            fprintf('In getNetLastActivations, net output dimension is %d\n', dims{:});
        end
        if numel(dims)==1
            activations_out(:, inter)=final_resp(:);
        elseif numel(dims)==2
            % if an error occurs below, probably batchSize==1. It's not
            % supported except the last layer is fisher vector.
            activations_out(:,:, inter)=final_resp;
        elseif numel(dims)==3
            activations_out(:,:,:, inter)=final_resp;
        else
            error('too many dim')
        end
    end
    activations=squeeze(activations_out);
end




