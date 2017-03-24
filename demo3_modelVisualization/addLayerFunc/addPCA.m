function net=addPCA(net, lastNchannel, projDim, use448, dataset, network, batchSize, learnW)
    pcaOut=floor(sqrt(projDim));
    assert(pcaOut^2==projDim);
    % pca is a filter of size: 1*1*lastNchannel*sqrtProjDim
    % with a bias of length sqrtProjDim. All in single. 
        
    pcaWeightFile=consts(dataset, 'pcaWeight',...
        'network', network, ...
        'cfgId', 'pca_fb', ...
        'projDim', projDim, ...
        'use448', use448);
    
    if exist(pcaWeightFile, 'file') == 2
        fprintf('Load PCA weight from saved file: %s\n', pcaWeightFile);
        load(pcaWeightFile);
    else
        % get activations from the last conv layer % checkpoint
        [trainFV, trainY, valFV, valY]=...
            get_activations_dataset_network_layer(...
            dataset, network, 'pcaConv', use448, net, batchSize);
       
        samples=permute(trainFV, [1,2,4,3]); % from hwcn to hwnc
        samples=reshape(samples, [], size(samples, 4));
        ave=mean(samples, 1); % 1*dim vector
        coeff=pca(samples); % dim*dim matrix, each column is a principle direction
        bias=-ave*coeff; % a row vector, should be the initial value for bias
        
        coeff=single(coeff); 
        bias=single(bias);
        
        savefast(pcaWeightFile, 'coeff', 'bias');
        fprintf('save PCA weight to new file: %s\n', pcaWeightFile);
    end
    initPCAparam={{reshape(coeff(:, 1:pcaOut), 1, 1, lastNchannel, pcaOut),...
                   bias(1:pcaOut)}};
    
    net.layers{end+1} = struct('type', 'conv', 'name', 'pca_reduction', ...
       'weights', initPCAparam, ...
       'stride', 1, ...
       'pad', 0, ...
       'learningRate', [1 2]*learnW);
end